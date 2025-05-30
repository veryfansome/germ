
def build_vocab(texts):
    """
    Build a character vocabulary from a list of strings.
    Returns:
      char2idx (dict)
      idx2char (list)
    """
    vocab = set()
    for t in texts:
        for ch in t:
            vocab.add(ch)
    # Add special tokens for mask, start-of-sequence, end-of-sequence if needed
    # We reserve index 0 for padding
    special_tokens = ["[PAD]", "[MASK]", "[UNK]"]
    vocab = sorted(list(vocab))
    idx2char = special_tokens + vocab
    char2idx = {ch: i for i, ch in enumerate(idx2char)}
    return char2idx, idx2char


long_text_examples = [
    ("Sherlock Holmes took his bottle from the corner of the mantel-piece and his hypodermic syringe from its "
     "neat morocco case. With his long, white, nervous fingers he adjusted the delicate needle and rolled back "
     "his left shirt-cuff. For some little time his eyes rested thoughtfully upon the sinewy forearm and wrist "
     "all dotted and scarred with innumerable puncture-marks."),
    # More!,
]

punctuation_chars = [".", ",", "!", "?", ";", ":"]

short_text_examples = [
    "Hi",
    "Hello world!",
    "You're the worst!",
    "What the hell?",
    "Where did you get that?",
    "This is a sample text.",
    "Another example text for multi-task learning.",
    # More!,
]

char2idx, idx2char = build_vocab(
    punctuation_chars
    + short_text_examples
    + short_text_examples
)
