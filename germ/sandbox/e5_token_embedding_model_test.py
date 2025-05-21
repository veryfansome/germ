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
probes = ["I", "a", "bank", "book", "cat", "run", "democracy", "neural", "quantum"]

# --- load model ------------------------------------------------------
embedder = TokenEmbeddingModel(device=device)
embedder.load_state_dict(torch.load(ckpt, map_location=device))
embedder.eval()

# --- build hold-out sample ------------------------------------------
vocab = json.load(open("data/e5_token_embedding_model/vocab.json"))          # save the list returned by build_vocab()
holdout = [w for w in vocab if w not in probes]
random.shuffle(holdout)
#holdout = holdout[:3000]

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

# --- neighbour inspection -------------------------------------------
for w in probes:
    v = embedder([w]).detach().cpu().numpy().astype("float32")
    faiss.normalize_L2(v)
    d, idx = index.search(v, 8)
    nbrs = [holdout[i] for i in idx[0]]
    print(f"{w:>10} →", ", ".join(f"{n}:{s:.2f}" for n, s in zip(nbrs, d[0])))

# --- AUROC with WordNet synonyms ------------------------------------
pairs = []
labels = []
for w in probes:
    syns = {l.name() for s in wn.synsets(w) for l in s.lemmas()} & set(holdout)
    for s in random.sample(list(syns), min(20, len(syns))):
        pairs.append((w, s))
        labels.append(1)
    for _ in range(20):
        r = random.choice(holdout)
        pairs.append((w, r))
        labels.append(0)

vec_a = embedder([p[0] for p in pairs])
vec_b = embedder([p[1] for p in pairs])
cos = torch.cosine_similarity(vec_a, vec_b).detach().cpu().numpy()
print("Word-level synonym AUROC =", roc_auc_score(labels, cos))
