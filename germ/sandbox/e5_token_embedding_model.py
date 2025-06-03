import collections
import json
import math
import nltk
import numpy as np
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from nltk.util import ngrams as nltk_ngrams
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from typing import Iterable, List

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords as nltk_stop, wordnet as wn

"""
* Components -------------------------------------------------------------

  • **TokenEmbeddingModel**
      – wraps **intfloat/e5-base-v2** and adds a linear *token head* that
        produces order‑aware, L2‑normalised embeddings for *single words*.

  • **fine_tune_token_head()** (scriptable entry point)
      – self‑supervised contrastive training on WikiText‑103 to warm‑up the
        token head before the controller is trained.
"""


def build_antonym_map(vocab_tokens: list[str]) -> dict[str, list[str]]:
    """
    Return { token : [ antonym₁, antonym₂, … ] } restricted to *tokens that
    are themselves in vocab_tokens*.  Uses NLTK WordNet.
    """
    antonym_dict: dict[str, set[str]] = collections.defaultdict(set)
    key_to_tokens: dict[str, list[str]] = collections.defaultdict(list)
    token_meta: dict[str, dict[str, list[int] | str]] = {}
    for t in vocab_tokens:
        key_chars, blank_pos = [], []
        for i, ch in enumerate(t):
            if ch in {" ", "-"}:
                key_chars.append("_")
                blank_pos.append(len(key_chars) - 1)
            else:
                key_chars.append(ch)
        key = "".join(key_chars)  # WordNet canonical key
        token_meta[t] = {"key": key, "positions": blank_pos}
        key_to_tokens[key].append(t)
    for key, surface_tokens in key_to_tokens.items():
        for syn in wn.synsets(key):
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    ant_key = ant.name() # already underscore form
                    if ant_key not in key_to_tokens:
                        continue  # antonym not in our vocab
                    # For every surface spelling that maps to ant_key
                    for ant_tok in key_to_tokens[ant_key]:
                        for src_tok in surface_tokens:
                            if ant_tok != src_tok:
                                antonym_dict[src_tok].add(ant_tok)
                                antonym_dict[ant_tok].add(src_tok)  # symmetric
    return {k: sorted(v) for k, v in antonym_dict.items()}


def build_stop_words():
    stop_words = set()
    stop_words_to_prune = set()
    for w in nltk_stop.words("english"):
        if "'" in w:
            pair = w.split("'")
            pair[-1] = "'" + pair[-1]
            if pair[-1] == "'t" and pair[0].endswith("n") and pair[0] not in {"ain", "can", "shan", "won"}:
                stop_words_to_prune.add(pair[0])
                pair[0] = pair[0][:-1]
                pair[-1] = "n" + pair[-1]
            stop_words.update(pair)
        elif len(w) == 1 and w not in {"a", "i", "o", "y"}:
            stop_words.add("'" + w)
        else:
            stop_words.add(w)
    stop_words.difference_update(stop_words_to_prune)
    return stop_words


def build_synonym_map(vocab_tokens: list[str]) -> dict[str, list[str]]:
    """
    { token : [ synonym₁, synonym₂, … ] }  limited to the vocab.
    """
    synonym_dict: dict[str, set[str]] = collections.defaultdict(set)
    key_to_tokens: dict[str, list[str]] = collections.defaultdict(list)

    def canon(t: str) -> str:
        return t.replace(" ", "_").replace("-", "_")

    for t in vocab_tokens:
        key_to_tokens[canon(t)].append(t)

    for key, surface_tokens in key_to_tokens.items():
        for syn in wn.synsets(key):
            for lemma in syn.lemmas():
                syn_key = lemma.name()
                if syn_key not in key_to_tokens:
                    continue
                for dst in key_to_tokens[syn_key]:
                    for src in surface_tokens:
                        if dst != src:
                            synonym_dict[src].add(dst)
                            synonym_dict[dst].add(src)
    # drop items with >25 syns (WordNet artefacts like “run”)
    return {k: sorted(v) for k, v in synonym_dict.items() if len(v) <= 25}


def extract_ngrams(
        tokens: List[str],
        stop_words: set[str],
        ngram_min: int = 2,
        ngram_max: int = 7
) -> Iterable[str]:
    """
    Yield space-joined n-grams from *tokens* for all n in [ngram_min, ngram_max].
    """
    if not tokens:
        return  # nothing to yield
    max_n = min(len(tokens), ngram_max)
    for n in range(ngram_min, max_n + 1):
        for gram in nltk_ngrams(tokens, n):
            if gram[0].lower() in stop_words or gram[-1].lower() in stop_words:
                continue
            yield " ".join(gram)


def group_by_consecutive_keys(data):
    if not data:
        return []

    # Sort the dictionary keys
    sorted_keys = sorted(data.keys())

    grouped_values = []
    current_group = [data[sorted_keys[0]]]

    # Track previous key to detect consecutive runs
    previous_key = sorted_keys[0]
    for key in sorted_keys[1:]:
        # If the current key is consecutive to the previous key,
        # append the corresponding value to the current group
        if key == previous_key + 1:
            current_group.append(data[key])
        else:
            # If not consecutive, start a new group
            grouped_values.append(current_group)
            current_group = [data[key]]
        previous_key = key

    # Don't forget to add the last group
    grouped_values.append(current_group)
    return grouped_values


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Pure out-of-place L2 normalisation.
    No in-place ops -> safe for autograd on all back-ends.
    """
    denom = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / denom


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)  # covers CUDA & MPS
    torch.backends.cudnn.benchmark = False


class TokenEmbeddingModel(nn.Module):
    """Wraps **intfloat/e5-base-v2** and adds a projection head so that
    *individual words* (tokens) get a reusable, order‑aware embedding.

    The base encoder is optionally frozen for cheap fine‑tuning.
    """

    def __init__(
            self,
            base_model_name: str = "intfloat/e5-large-v2",
            device: torch.device | str = "cpu",
            embed_dim: int = 2048,  # 2x e5-large-v2's 1024 dimensions
            freeze_base: bool = True,
            proj_hidden: int = 3072,  # embed_dim x 1.5
    ):
        super().__init__()
        self.device = torch.device(device)

        # frozen base encoder
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name).to(self.device)
        if freeze_base:
            for p in self.encoder.parameters():
                p.requires_grad = False
                self.encoder.eval()

        # projection head (encoder → token embed)
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, proj_hidden),
            nn.GELU(),
            nn.LayerNorm(proj_hidden),
            nn.Linear(proj_hidden, embed_dim, bias=False),
        ).to(self.device)

        # reconstruction head (token embed → encoder)
        self.reconstruct = nn.Sequential(
            nn.Linear(embed_dim, proj_hidden),
            nn.GELU(),
            nn.LayerNorm(proj_hidden),
            nn.Linear(proj_hidden, self.encoder.config.hidden_size, bias=False),
        ).to(self.device)

        for m in list(self.proj.modules()) + list(self.reconstruct.modules()):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)

        self._embed_dim = embed_dim

    @property
    def embed_dim(self):
        return self._embed_dim

    def encode_with_hidden(self, texts: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return **L2‑normalised** embeddings for *texts* (list of words)."""
        batch = self.tokenizer(
            #[f"passage: {t}" for t in texts],
            [f"query: {t}" for t in texts],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        hidden = self.encoder(**batch).last_hidden_state[:, 0, :]  # CLS
        vec = F.normalize(self.proj(hidden), p=2, dim=-1)
        return vec, hidden

    def forward(self, texts: List[str]) -> torch.Tensor:
        vec, _ = self.encode_with_hidden(texts)
        return vec

    def train(self, mode: bool = True):
        super().train(mode)
        return self

class VocabDataset(Dataset):
    """A simple list‑of‑tokens dataset built from WikiText‑103."""

    def __init__(self, tokens: List[str]):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


def build_vocab(
        ngram_max: int = 7,
        ngram_min: int = 2,
) -> (List[str], List[str]):
    """Extract a vocabulary from *WikiText‑103‑raw‑v1* via datasets."""
    print(f"Loading dataset")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    ds_size = len(ds)
    print(f"Selecting training vocabulary from {ds_size} documents")

    all_lowercase_token_pattern = re.compile(r"^(?:[a-z]+')?[a-zàáçćèéïōšüū-]+$")
    all_uppercase_token_pattern = re.compile(r'^(?:[A-Z]\.?)+$')
    ain_can_shan_won_pattern = re.compile(r"^(?:[Aa]in|[Cc]an|[Ss]han|[Ww]on)$")
    number_pattern = re.compile(r"^([0-9]+\.?[0-9]*|[1-9][0-9]{,2}(?:,[0-9]{3})*(\.[0-9]*)?)$")
    numeric_pattern = re.compile(r"^(?:[a-z]+-)?"
                                 r"(?:[0-9]+\.?[0-9]*|[1-9][0-9]{,2}(?:,[0-9]{3})*(\.[0-9]*)?)"
                                 r"(?:[a-zA-Z]+)?(?:-[a-zA-Z]+)*?$")
    starts_with_uppercase_pattern = re.compile(r"^(?:[a-zàáçćèéïōšüū]+[-']?)?[A-ZÁÅÆÉĐÍÎÓŠ]")
    stop_words = build_stop_words()

    all_lowercase_token_counter = collections.Counter()
    anomalous_token_counter = collections.Counter()
    named_entity_counter = collections.Counter()
    ngram_counter = collections.Counter()
    number_counter = collections.Counter()
    numeric_counter = collections.Counter()

    for row_id, row in enumerate(ds):
        tokens = row["text"].split()

        preprocessed_tokens = []
        concat_into_previous = False
        for token in tokens:
            if concat_into_previous:
                preprocessed_tokens[-1] += token
                concat_into_previous = False
            elif token.startswith("'"):
                if not preprocessed_tokens:
                    preprocessed_tokens.append(token)
                elif preprocessed_tokens[-1].endswith("n") and token == "'t":
                    if bool(ain_can_shan_won_pattern.search(preprocessed_tokens[-1])):
                        preprocessed_tokens.append("'t")
                    else:
                        preprocessed_tokens[-1] = preprocessed_tokens[-1][:-1]
                        preprocessed_tokens.append("n't")
                elif token not in {"'", "'d", "'ll", "'m", "'re", "'s", "'ve"}:
                    preprocessed_tokens[-1] += token
                else:
                    preprocessed_tokens.append(token)
            elif token in {"@,@", "@.@", "@-@"}:
                preprocessed_tokens[-1] += token.strip("@")
                concat_into_previous = True
            else:
                preprocessed_tokens.append(token)

        all_lowercase_tokens = {}
        anomalous_tokens = {}
        capitalized_tokens = {}
        number_tokens = {}
        numeric_tokens = {}

        for token_idx, token in enumerate(preprocessed_tokens):
            if bool(all_lowercase_token_pattern.search(token)):
                all_lowercase_tokens[token_idx] = token
            elif bool(all_uppercase_token_pattern.search(token)):
                capitalized_tokens[token_idx] = token
            elif bool(starts_with_uppercase_pattern.search(token)):
                # TODO:
                #   - connected ngrams
                capitalized_tokens[token_idx] = token
            elif bool(number_pattern.search(token)):
                number_tokens[token_idx] = token
            elif bool(numeric_pattern.search(token)):
                numeric_tokens[token_idx] = token
            else:
                anomalous_tokens[token_idx] = token

        if all_lowercase_tokens:
            all_lowercase_token_counter.update(all_lowercase_tokens.values())
            all_lowercase_ngrams = []
            for token_group in group_by_consecutive_keys(all_lowercase_tokens):
                all_lowercase_ngrams.extend(
                    extract_ngrams(token_group, stop_words, ngram_min=ngram_min, ngram_max=ngram_max)
                )
            ngram_counter.update(all_lowercase_ngrams)

        if anomalous_tokens:
            anomalous_token_counter.update(anomalous_tokens.values())

        if capitalized_tokens:
            named_entity_counter.update([t for t in capitalized_tokens.values()
                                         if t.lower() not in all_lowercase_token_counter])
            capitalized_ngrams = []
            for token_group in group_by_consecutive_keys(capitalized_tokens):
                capitalized_ngrams.extend(
                    extract_ngrams(token_group, stop_words, ngram_min=ngram_min, ngram_max=ngram_max)
                )
            named_entity_counter.update(capitalized_ngrams)

        if number_tokens:
            number_counter.update(number_tokens.values())

        if numeric_tokens:
            numeric_counter.update(numeric_tokens.values())

        if row_id > 0 and row_id % 100000 == 0:
            print(f"Processed {row_id} rows")

    corpus = {}
    for k, v in {
        "all_lowercase_token": (all_lowercase_token_counter, 100),
        "anomalous_token": (anomalous_token_counter, 100),
        "named_entity": (named_entity_counter, 100),
        "ngram": (ngram_counter, 100),
        "number": (number_counter, 100),
        "numeric": (numeric_counter, 100),
    }.items():
        corpus[k] = [item for item, c in v[0].items() if c >= v[1]]
        corpus[k] = sorted(corpus[k], key=v[0].get, reverse=True)[:100000]
        random.shuffle(corpus[k])
        print(f"{k}: {len(corpus[k])}")
    return corpus


def fine_tune_token_head(
        accum_steps: int = 4,  # ← batch_size × accum_steps = “virtual batch”
        #accum_steps: int = 8,
        device: str | None = None,
        epochs: int = 4,
        #epochs: int = 8,
        #epochs: int = 16,
        #epochs: int = 24,
        #epochs: int = 32,
        #epochs: int = 40,
        #batch_size: int = 256,
        batch_size: int = 512,
        lr: float = 1e-5,
        out_dir: str = "data/e5_token_embedding_model",
        recon_max_weight: float = 1.0,  # weight AFTER warm-up
        recon_warmup_epochs: int = 3,  # how many epochs to ramp 0 → max
        seed: int = 42,
        sem_weight: float = 0.2,  # weight for semantic-anchor loss
        syn_weight: float = 0.1,  # weight for synonym loss

):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    if device is None:
        device = (
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
    device_t = torch.device(device)

    # Each training loop iteration processes a batch of tokens. The training objectives are:
    # - Match each word with itself:
    #     Each word's embedding is compared to itself, ensuring that the representation of a
    #     word remains the same. This “positive” pair (word with itself) is ideally the most similar.
    # - Differentiate each word from other words:
    #     Find the most "confusing" other word in the batch — i.e., a word whose embedding is most similar (even though
    #     it’s not supposed to be the same). This confusing word becomes the “negative” example. The model is then
    #     penalized if it mixes up these representations.
    antonym_file = Path(out_dir) / "antonyms.json"
    corpus_file = Path(out_dir)/"corpus.json"
    synonym_file = Path(out_dir) / "synonyms.json"
    if antonym_file.exists() and corpus_file.exists() and synonym_file.exists():
        with antonym_file.open() as f:
            antonyms: dict[str, list[str]] = json.load(f)
        with corpus_file.open("r", encoding="utf-8") as f:
            corpus = json.load(f)
        with synonym_file.open("r", encoding="utf-8") as f:
            synonyms = json.load(f)
    else:
        corpus = build_vocab()
        with corpus_file.open("w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=4)
        for k, v in corpus.items():
            with (Path(out_dir)/f"{k}.json").open("w", encoding="utf-8") as f:
                json.dump(v, f, ensure_ascii=False, indent=4)
        antonyms = build_antonym_map(corpus["all_lowercase_token"] + corpus["ngram"])
        with antonym_file.open("w") as f:
            json.dump(antonyms, f, ensure_ascii=False, indent=4)
        synonyms = build_synonym_map(corpus["all_lowercase_token"] + corpus["ngram"])
        with synonym_file.open("w") as f:
            json.dump(synonyms, f, ensure_ascii=False, indent=4)
    dataset = VocabDataset([
        item for k, sublist in corpus.items() for item in sublist if k not in {
            "anomalous_token",
            "number",
        }
    ])
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(dataset, batch_size=batch_size, generator=g, shuffle=True, num_workers=0, collate_fn=lambda b: b)

    model = TokenEmbeddingModel(freeze_base=True, device=device_t).to(device_t)
    optimiser = torch.optim.AdamW(list(model.proj.parameters()) + list(model.reconstruct.parameters()),
                                  lr=lr, weight_decay=1e-2)

    # scheduler

    # A temperature parameter (tau) is used to adjust the sensitivity of similarity comparisons. It starts higher and
    # is annealed to a lower value during training. This helps fine-tune how sharply the model distinguishes between
    # the correct word and its hardest negative counterpart.
    base_tau, final_tau = 0.075, 0.04
    recon_warmup_steps = recon_warmup_epochs * len(dl)
    total_iter = epochs * len(dl)

    scheduler = get_cosine_schedule_with_warmup(
        optimiser,
        int(0.05 * total_iter),  # warmup steps
        total_iter
    )

    # training loop

    model.train()
    optimiser.zero_grad(set_to_none=True)
    for ep in range(epochs):
        running_loss = 0.0
        for step, batch_tokens in enumerate(dl):
            iter_global = ep * len(dl) + step

            #  temperature & reconstruction-weight annealing
            tau = final_tau + 0.5 * (base_tau - final_tau) * (
                1.0 + math.cos(math.pi * iter_global / total_iter)
            )
            recon_weight = (
                recon_max_weight * iter_global / recon_warmup_steps
                if iter_global < recon_warmup_steps
                else recon_max_weight
            )

            # The model's `encode` method generates embeddings for all tokens in the batch. These embeddings are
            # L2-normalized since we use cosine similarity.
            embeds, base_hidden = model.encode_with_hidden(batch_tokens)  # (B,d)
            B, d = embeds.shape                                           # noqa: N806

            # For each anchor embedding, cosine similarities with all embeddings in the batch are computed.
            # The similarity with itself is ignored by filling it with a large negative value (-1e4). The hardest
            # negative embedding is selected for each anchor by identifying the index of the maximum similarity value
            # that is not from its own embedding.
            with torch.no_grad():
                sim = embeds @ embeds.T  # (B, B), cosine similarities
                # Manually zero-diag because fill_diagonal_ is flaky on MPS
                idx = torch.arange(B, device=sim.device)
                sim[idx, idx] = -1e4  # mask self, ignore self-sim
                _, hard_idx = sim.topk(k=2, dim=-1)  # hard_idx[:,0] = hardest, hard_idx[:,1] = 2nd-hardest
            hard_neg1 = embeds[hard_idx[:, 0]].detach()  # (B, d)  – hardest negative (for every row)
            hard_neg2 = embeds[hard_idx[:, 1]].detach()  # (B, d)  – 2nd-hardest (fallback pool)

            # Antonym negatives
            batch_antonyms, rows_with_antonyms = [], []
            for i, tok in enumerate(batch_tokens):
                if tok in antonyms:
                    antonym_candidate = random.choice(antonyms[tok])
                    if antonym_candidate == tok:  # WordNet sometimes returns identical lemma
                        continue  # fall back to hard_neg2
                    batch_antonyms.append(antonym_candidate)
                    rows_with_antonyms.append(i)
                else:
                    batch_antonyms.append(None)  # placeholder
            ant_embeds = hard_neg2.clone()  # start with fallback
            if rows_with_antonyms:  # encode real antonyms
                ant_strings = [batch_antonyms[i] for i in rows_with_antonyms]
                with torch.no_grad():
                    ant_vecs, _ = model.encode_with_hidden(ant_strings)
                ant_embeds[torch.tensor(rows_with_antonyms, device=embeds.device)] = ant_vecs

            # Synonym attraction loss
            rows_with_syn, syn_strings = [], []
            for i, tok in enumerate(batch_tokens):
                if tok in synonyms:
                    syn_strings.append(random.choice(synonyms[tok]))
                    rows_with_syn.append(i)
            if rows_with_syn:
                syn_vecs, _ = model.encode_with_hidden(syn_strings)  # keep grad ON
                # cosine distance (1 - cos)
                syn_sim = F.cosine_similarity(
                    embeds[torch.tensor(rows_with_syn, device=embeds.device)], syn_vecs, dim=-1
                )
                loss_syn = F.relu(0.95 - syn_sim).mean()  # only penalise if < margin of 0.95
            else:
                loss_syn = embeds.new_tensor(0.0)

            # Positives are the anchors themselves but with a *tiny* jitter so the similarity is < 1, and we still
            # get a gradient
            pos_noise = 1e-3 * torch.randn_like(embeds)
            z_pos = l2_normalize(embeds + pos_noise)   # (B, d)

            z_anchor = embeds  # (B,d)

            # Build candidate matrix
            candidates = torch.stack((z_pos, hard_neg1, ant_embeds), dim=1)  # (B, 3, d)
            candidates = candidates.reshape(-1, d)  # (3B, d)
            logits = (z_anchor @ candidates.T) / tau  # (B, 3B)
            labels = torch.arange(B, device=device_t) * 3  # each row’s positive is at column i*3

            # Contrastive loss is computed using a cross-entropy style setup. For each element in the batch, positive
            # examples are the embeddings themselves, while the hardest negatives are pooled. Loss is scaled down by
            # `accum_steps` # to support gradient accumulation.
            loss_contrast = F.cross_entropy(logits, labels, reduction="mean")

            pred_hidden = model.reconstruct(embeds)
            loss_recon = F.mse_loss(pred_hidden, base_hidden.detach())

            # Compute cosine-sim matrices in each space, then try to match them. Nothing in loss_recon forces the
            # geometry of the new embedding space to remain correlated with the original sentence-level geometry.
            # Anchor it explicitly so we can have global semantic consistency.
            with torch.no_grad():  # (B,B) backbone similarities
                back_sim = F.cosine_similarity(
                    base_hidden.unsqueeze(1), base_hidden.unsqueeze(0), dim=-1
                )
            proj_sim = F.cosine_similarity(
                embeds.unsqueeze(1), embeds.unsqueeze(0), dim=-1
            )
            loss_sem = F.mse_loss(proj_sim, back_sim)  # zero-grad flows only thru proj_sim

            # To encourage diversity, a regularization term is added to the loss. It penalizes the overlap
            # (non-orthogonality) between the learned projections in the MLP head's final layer, to keep the different
            # dimensions as independent as possible. This means the model is nudged to learn representations where
            # features don’t overlap too much, making them more robust and less redundant.
            #W = F.normalize(model.proj[-1].weight, dim=1)
            #loss += 1e-4 * (W @ W.T).pow(2).triu(1).mean() / accum_steps

            # Gradients are computed for parameter updates. A gradient accumulation strategy is employed every
            # `accum_steps` micro-steps to form a "virtual batch".
            loss = (loss_contrast +
                    recon_weight * loss_recon +
                    sem_weight * loss_sem +
                    syn_weight * loss_syn) / accum_steps
            loss.backward()
            running_loss += loss.item() * accum_steps  # un-scale for logging

            # Gradients are clipped to prevent exploding gradients, the optimizer updates the model's parameters, and
            # the learning rate schedule steps forward.
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # optional
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                scheduler.step()  # one LR-update per *optim* step, not per micro-step

            if step % 50 == 0:
                with torch.no_grad():
                    pred_to_tgt_sim = F.cosine_similarity(
                        pred_hidden, base_hidden, dim=-1
                    ).mean()
                    print(
                        f"epoch {ep+1}/{epochs}  "
                        f"step {step:04d}/{len(dl)}  "
                        f"τ={tau:.4f}  "
                        f"loss={running_loss/(step+1):.4f}  "
                        f"contrast={loss_contrast.item():.4f}  "
                        f"recon={loss_recon.item():.4f}  "
                        f"sem={loss_sem.item():.4f}  "
                        f"syn={loss_syn.item():.4f}  "
                        f"⟨cos(pred, tgt)⟩={pred_to_tgt_sim.item():.3f}"
                    )
        print(f"→ epoch {ep+1} mean loss {running_loss/len(dl):.4f}")

    torch.save(model.state_dict(), Path(out_dir)/"token_head.pt")
    print(f"Saved fine‑tuned TokenEmbeddingModel to {out_dir}/token_head.pt")


if __name__ == "__main__":
    fine_tune_token_head()
