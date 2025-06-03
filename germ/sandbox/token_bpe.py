from collections import defaultdict, Counter


def get_vocab(corpus):
    """
    Build initial vocabulary of word-character sequences from the corpus.
    Each word is split into individual characters, plus a special </w> symbol at the end.
    """
    vocab = {}
    for word in corpus:
        chars = list(word) + ["</w>"]
        token = " ".join(chars)
        vocab[token] = vocab.get(token, 0) + 1
    return vocab


def get_pair_counts(vocab):
    """
    Count the frequency of each adjacent pair in the current vocabulary.
    """
    pair_counts = defaultdict(int)
    for token, freq in vocab.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            pair_counts[(symbols[i], symbols[i + 1])] += freq
    return pair_counts


def merge_vocab(pair_to_merge, vocab):
    """
    Merge all instances of the most frequent pair in the vocabulary.
    """
    merged_vocab = {}
    bigram = " ".join(pair_to_merge)
    replacement = "".join(pair_to_merge)

    for token, freq in vocab.items():
        # Replace the pair with its merged form
        merged_token = token.replace(bigram, replacement)
        merged_vocab[merged_token] = merged_vocab.get(merged_token, 0) + freq

    return merged_vocab


def train_bpe(corpus, num_merges=10):
    """
    Train a BPE tokenizer:
    1. Initialize the vocabulary.
    2. Iteratively merge frequent pairs.
    3. Return the merges and final vocabulary.
    """
    vocab = get_vocab(corpus)
    merges = []

    for _ in range(num_merges):
        pair_counts = get_pair_counts(vocab)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        merges.append(best_pair)
        vocab = merge_vocab(best_pair, vocab)

    return merges, vocab


def apply_bpe(word, merges):
    """
    Given a word and a list of merges, apply BPE operations to produce subword tokens.
    """
    # Start by splitting into characters plus the </w> end symbol
    symbols = list(word) + ["</w>"]

    # Repeatedly merge where possible
    i = 0
    while i < len(symbols) - 1:
        # Check if the pair is a known merge
        pair = (symbols[i], symbols[i + 1])
        if pair in merges:
            # Merge them
            symbols[i:i + 2] = ["".join(pair)]
            i = max(i - 1, 0)
        else:
            i += 1

    # Remove the end-of-word symbol for final subword tokens
    if symbols[-1] == "</w>":
        symbols = symbols[:-1]
    return symbols


if __name__ == "__main__":
    # Example corpus of words
    example_corpus = ["low", "lowest", "newer", "wider"]

    # Train the BPE tokenizer
    num_merges = 10
    merges, final_vocab = train_bpe(example_corpus, num_merges)
    merges_set = set(merges)  # Convert merges list to a set for faster lookup

    # Encode a sample word
    word_to_encode = "lower"
    encoded = apply_bpe(word_to_encode, merges_set)

    print("Learned merges:", merges)
    print("Encoded word:", word_to_encode, "->", encoded)