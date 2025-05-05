import aiohttp
import asyncio
import numpy as np
import random
import re
from collections import Counter
from datasets import load_dataset


async def load_gutenberg(book_id):
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()  # raises exception if not 2xx
            chunks = []
            async for chunk in response.content.iter_chunked(1024):
                chunks.append(chunk.decode("utf-8", errors="replace"))
            return "".join(chunks)


def build_vocab(texts, min_percentile=1):
    """
    Build a character vocabulary from a list of strings, excluding characters in the lowest frequency percentile.
    Returns:
      char2idx (dict)
      idx2char (list)
    """
    # Count the frequency of each character
    counter = Counter(ch for t in texts for ch in t)
    # Calculate frequency percentiles
    frequencies = np.array(list(counter.values()))
    thresholds = np.percentile(frequencies, min_percentile)
    # Filter out characters below the given frequency percentile
    vocab = [ch for ch, freq in counter.items() if freq > thresholds]
    # Add special tokens for mask, start-of-sequence, end-of-sequence if needed
    special_tokens = ["[PAD]", "[MASK]", "[UNK]"]
    vocab = sorted(vocab)
    idx2char = special_tokens + vocab
    char2idx = {ch: i for i, ch in enumerate(idx2char)}
    return char2idx, idx2char


def build_wikitext_short_text_corpus(max_len: int = 256, split: str = "train"):
    """Extract a small corpus of clean short-form text from WikiText‑103‑raw‑v1 for encoder testing."""
    print(f"Loading wikitext dataset")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    ds_size = len(ds)
    print(f"Selecting {split} corpus from {ds_size} wikitext documents")
    candidates = []
    startswith_uppercase_pattern = re.compile(r"[A-Z]")
    for row_id, row in enumerate(ds):
        row_text = row["text"].strip()
        if (
                # Filters to reduce noise.
                not row_text or not row_text.isascii()  # Exclude non-english characters
                or not bool(startswith_uppercase_pattern.match(row_text))  # Must start with uppercase
                or row_text[-1] not in {".", "?", "!"}  # Must have standard punctuation
                or len(row_text) < 7  # At least 7 words
        ):
            continue
        if len(row_text) <= max_len:
            candidates.append(row_text)
        if row_id > 0 and row_id % 250000 == 0:
            print(f"Processed {row_id} rows")
    print(f"Selected {len(candidates)} {split} candidates from wikitext")
    for c in random.sample(candidates, 10):
        print(f" => {c}")
    return candidates


def build_wikitext_word_corpus() -> dict[str, list[str]]:
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

    all_lowercase_token_counter = Counter()
    anomalous_token_counter = Counter()
    named_entity_counter = Counter()
    number_counter = Counter()
    numeric_counter = Counter()

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
                capitalized_tokens[token_idx] = token
            elif bool(number_pattern.search(token)):
                number_tokens[token_idx] = token
            elif bool(numeric_pattern.search(token)):
                numeric_tokens[token_idx] = token
            else:
                anomalous_tokens[token_idx] = token

        if all_lowercase_tokens:
            all_lowercase_token_counter.update(all_lowercase_tokens.values())

        if anomalous_tokens:
            anomalous_token_counter.update(anomalous_tokens.values())

        if capitalized_tokens:
            named_entity_counter.update([t for t in capitalized_tokens.values()])

        if number_tokens:
            number_counter.update(number_tokens.values())

        if numeric_tokens:
            numeric_counter.update(numeric_tokens.values())

        if row_id > 0 and row_id % 250000 == 0:
            print(f"Processed {row_id} rows")

    corpus = {}
    for k, v in {
        "all_lowercase_token": (all_lowercase_token_counter, 100),
        "anomalous_token": (anomalous_token_counter, 100),
        "named_entity": (named_entity_counter, 100),
        "number": (number_counter, 100),
        "numeric": (numeric_counter, 100),
    }.items():
        corpus[k] = [item for item, c in v[0].items() if c >= v[1]]
        random.shuffle(corpus[k])
        print(f"{k}: {len(corpus[k])}")
    return corpus


long_text_examples = [
    ("Sherlock Holmes took his bottle from the corner of the mantel-piece and his hypodermic syringe from its "
     "neat morocco case. With his long, white, nervous fingers he adjusted the delicate needle and rolled back "
     "his left shirt-cuff. For some little time his eyes rested thoughtfully upon the sinewy forearm and wrist "
     "all dotted and scarred with innumerable puncture-marks."),
    # More!,
]


if __name__ == "__main__":
    #build_wikitext_corpus()
    print(asyncio.run(load_gutenberg(1342))[:6000])
