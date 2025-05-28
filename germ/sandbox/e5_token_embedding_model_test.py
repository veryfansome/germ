import faiss
import json
import nltk
import numpy as np
import random
import time
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score

from germ.sandbox.e5_token_embedding_model import TokenEmbeddingModel

nltk.download('wordnet')
from nltk.corpus import wordnet as wn


# --- config ----------------------------------------------------------
ckpt = Path("data/e5_token_embedding_model/token_head.pt")
device = "mps"
probes = [
    "🙂", ":)",
    "1492", "1776",
    "3.1415926",
    "49er", "mp3", "R2-D2", "sha1",  # alphanumerics
    "H₂O", "NaCL",
    "FBI", "FDIC",
    "I", "a", "and", "of", "the",
    "AAPL", "BRK.B", "GOOG",
    "Apple Inc", "apple",
    "San Francisco", "New York",
    "Bill", "George", "Nancy",
    "bank", "bass", "seal",  # homonyms
    "book", "cat", "democracy", "neural", "quantum",
    "buy", "purchase",  # synonyms
    "city", "cities", "run", "running", "ran",  # morphology
    "cold", "hot",  # antonyms
    "fuck", "shit",  # profanity
    "due diligence", "machine learning",  # ngrams
    "bought the farm", "kick the bucket", "red herring", "sleeps with the fishes",  # idioms
]

# --- load model ------------------------------------------------------
embedder = TokenEmbeddingModel(device=device)
embedder.load_state_dict(torch.load(ckpt, map_location=device))
embedder.eval()

# --- build hold-out sample ------------------------------------------
# NOTE: don't have to load and index everything the model's seen
named_entities = json.load(open("data/e5_token_embedding_model/named_entity.json"))
ngrams = json.load(open("data/e5_token_embedding_model/ngram.json"))
#numbers = json.load(open("data/e5_token_embedding_model/number.json"))
numerics = json.load(open("data/e5_token_embedding_model/numeric.json"))
vocab = json.load(open("data/e5_token_embedding_model/all_lowercase_token.json"))
holdout = [w for w in named_entities if w not in probes]
holdout.extend([w for w in ngrams if w not in probes])
#holdout.extend([w for w in numbers if w not in probes])
holdout.extend([w for w in numerics if w not in probes])
holdout.extend([w for w in vocab if w not in probes])
random.shuffle(holdout)

batch_size = 1000  # Adjust based on memory constraints
vecs_list = []

print("Embedding holdout vocab in batches")
for i in range(0, len(holdout), batch_size):
    batch_start_ts = time.time()
    batch = holdout[i:i + batch_size]
    vecs_batch = embedder(batch).detach().cpu().numpy().astype("float32")
    vecs_list.append(vecs_batch)
    print(f"Batch {i} took {time.time() - batch_start_ts}")

vecs = np.vstack(vecs_list)  # Combine all batch vectors into a final array

print("Normalizing holdout vectors")
faiss.normalize_L2(vecs)
index = faiss.IndexFlatIP(embedder.embed_dim)
print("Indexing holdout vectors")
index.add(vecs)

##
# Neighbour sanity-check

for w in probes:
    v = embedder([w]).detach().cpu().numpy().astype("float32")
    faiss.normalize_L2(v)
    d, idx = index.search(v, 12)
    nbrs = ", ".join(f"{holdout[j]}:{d[0,k]:.2f}" for k,j in enumerate(idx[0]))
    print(f"{w:>24} → {nbrs}")

##
# WordNet synonym AUROC – projection space *and* backbone CLS space

pairs, labels = [], []
for w in probes:
    syns = {l.name() for s in wn.synsets(w) for l in s.lemmas()} & set(holdout)
    for s in random.sample(list(syns), min(20, len(syns))):
        pairs.append((w, s))
        labels.append(1)
    for _ in range(20):
        r = random.choice(holdout)
        pairs.append((w, r))
        labels.append(0)

embed_a, hid_a = embedder.encode_with_hidden([a for a,_ in pairs])
embed_b, hid_b = embedder.encode_with_hidden([b for _,b in pairs])

cos_embed  = torch.cosine_similarity(embed_a, embed_b).detach().cpu().numpy()
cos_cls  = torch.cosine_similarity(hid_a, hid_b).detach().cpu().numpy()

print("\nAUROC on WordNet synonym test")
print(f"  • projection space   : {roc_auc_score(labels, cos_embed):.3f}")
print(f"  • backbone CLS space : {roc_auc_score(labels, cos_cls):.3f}")

##
# Inverse-head quality

#with torch.no_grad():
#    pred_hid = embedder.reconstruct(embed_a.to(embedder.device))
#    if embedder.device.type == "mps":
#        torch.mps.synchronize()
#    cos_inv  = torch.cosine_similarity(pred_hid.detach().cpu(), hid_a.detach().cpu()).mean().item()
#
#print(f"\n⟨cos(pred_hidden, true_hidden)⟩ on probe set: {cos_inv:.3f}")

##
# Synthetic negation flips

pairs, labels = [], []
negation_probes = [
    "I like cats.", "I do not like cats.",
    "We ship tomorrow.", "We do not ship tomorrow."
]
for a, b in zip(negation_probes[::2], negation_probes[1::2]):
    pairs += [(a, a, 1), (a, b, 0), (b, b, 1)]
    labels += [1, 0, 1]

sents_a = [p[0] for p in pairs]
sents_b = [p[1] for p in pairs]

embed_a, hid_a = embedder.encode_with_hidden([p[0] for p in pairs])
embed_b, hid_b = embedder.encode_with_hidden([p[1] for p in pairs])

cos_embed = torch.cosine_similarity(embed_a, embed_b).detach().cpu().numpy()
cos_cls = torch.cosine_similarity(hid_a, hid_b).detach().cpu().numpy()

print("\nSentence-level AUROC (paraphrase+negation)")
print(f"  • projection space                : {roc_auc_score(labels, cos_embed):.3f}")
print(f"  • backbone CLS space              : {roc_auc_score(labels, cos_cls):.3f}")

# extra: delta similarity for negation pairs
neg_idx = [i for i, p in enumerate(pairs) if " not " in p[1] or " not " in p[0]]
pos_idx = [i for i in range(len(pairs)) if i not in neg_idx]
print(f"  • Δsim(pos-pair) – Δsim(neg-pair) : {cos_embed[pos_idx].mean() - cos_embed[neg_idx].mean():.3f}")