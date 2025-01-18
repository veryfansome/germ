from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import numpy as np
import tensorflow as tf


def build_char_vocab():
    # 1) Start with a small set of special tokens
    # We'll keep '[PAD]' for padding, '[UNK]' for unknown
    char2idx = {"[PAD]": 0, "[UNK]": 1}
    idx2char = {0: "[PAD]", 1: "[UNK]"}

    # 2) Add printable ASCII (0x20 to 0x7E)
    #    This range includes letters, digits, punctuation
    ascii_printable_range = range(0x20, 0x7F)  # 32..126 inclusive
    current_index = 2

    for code_point in ascii_printable_range:
        ch = chr(code_point)
        if ch not in char2idx:  # just in case
            char2idx[ch] = current_index
            idx2char[current_index] = ch
            current_index += 1

    # 3) Add any extra sets you want, e.g. some basic Unicode punctuation:
    #    “ ” ‘ ’ — … etc., or certain emoji ranges, etc.
    extra_chars = ["…", "—", "“", "”", "‘", "’", "€", "£", "•"]
    for ch in extra_chars:
        if ch not in char2idx:
            char2idx[ch] = current_index
            idx2char[current_index] = ch
            current_index += 1

    # 4) If you want a few emojis, add them. The selection below should be thought of as emoji classes. Additional
    #    preprocessing can expand coverage of more emojis without bloating the embedding matrix. Convert similar emojis
    #    to one of the ones below.
    emojis = ["😄", "😞", "😡", "😜", "😂", "😍", "👍", "👎", "🎉"]
    for emo in emojis:
        if emo not in char2idx:
            char2idx[emo] = current_index
            idx2char[current_index] = emo
            current_index += 1

    return char2idx, idx2char


def corpus_to_sequences(corpus, char2idx, label2idx):
    x_list = []
    y_list = []

    for sentence_tokens in corpus:
        sentence_chars, sentence_labels = tokenize_chars_and_labels(sentence_tokens)

        x = [encode_char(c, char2idx) for c in sentence_chars]
        y = [label2idx[label] for label in sentence_labels]

        x_list.append(x)
        y_list.append(y)

    return x_list, y_list


def encode_char(ch, char2idx):
    """
    Convert a single character into its integer ID.
    If it's not found in char2idx, return the index for '[UNK]'.
    """
    if ch in char2idx:
        return char2idx[ch]
    else:
        return char2idx["[UNK]"]


def pad_sequences_custom(x_list, y_list, max_seq_len=None, pad_value=0):
    """
    Pad the sequences in X_list and Y_list to max_seq_len.
    If max_seq_len is None, use the longest sequence in X_list.

    Returns:
      X_padded: numpy array of shape (num_sentences, max_seq_len)
      Y_padded: numpy array of shape (num_sentences, max_seq_len)
    """
    if max_seq_len is None:
        max_seq_len = max(len(seq) for seq in x_list)

    # Initialize arrays
    x_padded = np.full((len(x_list), max_seq_len), pad_value, dtype='int32')
    y_padded = np.full((len(x_list), max_seq_len), pad_value, dtype='int32')

    # Fill them in
    for i, (x_seq, y_seq) in enumerate(zip(x_list, y_list)):
        seq_len = len(x_seq)
        x_padded[i, :seq_len] = x_seq
        y_padded[i, :seq_len] = y_seq

    return x_padded, y_padded


def tokenize_chars_and_labels(sentence_tokens):
    """
    For a single sentence (list of tokens), return:
      - list of characters (flattened across tokens)
      - list of B/I/E/S tags (aligned with each character)
    """
    chars = []
    labels = []

    for token in sentence_tokens:
        if len(token) == 1:
            # Single-character token => label is S
            chars.append(token[0])
            labels.append("S")
        else:
            # Multi-character token
            token_len = len(token)
            for i, c in enumerate(token):
                if i == 0:
                    chars.append(c)
                    labels.append("B")
                elif i == token_len - 1:
                    chars.append(c)
                    labels.append("E")
                else:
                    chars.append(c)
                    labels.append("I")
    return chars, labels


if __name__ == "__main__":
    _char2idx, _idx2char = build_char_vocab()

    _label2idx = {"B": 0, "I": 1, "E": 2, "S": 3}
    _idx2label = {v: k for k, v in _label2idx.items()}

    sample_corpus = [
        ["Hello", ",", "world", "!"],
        ["I", "love", "Python", "3.12"],
        ["Use", "a", "subnet", "of", "255.255.255.0", "or", "/24", "."],
        ["My", "router", "is", "located", "at", "192.168.1.1", "with", "a", "subnet", "of", "/23", "."],
        ["At", "my", "company", ",", "our", "internal", "network", "uses", "10.0.0.0/8"],
        ["An", "IP", "address", "might", "look", "like", "this", ":", "192.168.1.10", "."],
        ["127.0.0.1", "is", "an", "IPv4", "address", "."],
        ["The", "address", "127.0.0.1", "is", "often", "referred", "to", "as", "the", "loopback", "address", "."],
        ["The", "notation", "10.0.0.0/8", "is", "a", "representation", "of", "an", "IP", "address", "and", "its",
         "associated", "subnet", "mask", "in", "CIDR", "(", "Classless", " Inter-Domain", " Routing", ")",
         "notation", "."],
        ["Only", "192.168.x.x", "is", "."],
        ["The", "entire", "127.x.x.x", "address", "block", "is", "reserved", "for", "loopback", ",", "which",
         "seems", "like", "a", "colossal", "waste", "."],
        ["128.0.0.0", "is", "the", "start", "address", "of", "formerly", "\"", "Class", "B", "\"", "."],
        ["169.254.0.0/16", "(", "169.254.0.0", "–", "169.254.255.255", ")", "has", "been", "reserved", "for",
         "link-local", "addressing", "(", "RFC", "6890", ")", "."],
        ["55.255.255.255", "is", "reserved", "for", "limited", "broadcast", "destination", "address", "."],
        ["It", "allows", "for", "(", "2", "^", "{", "24", "}", "-", "2", ",", "or", "16,777,214", "usable", "addresses", "."],
        ["First", ",", "you", "need", "to", "`", "terraform", "init", "`", "."],
        ["To", "become", "the", "root", "user", ",", "you", "can", "run", "`", "sudo", "-i", "`", ",", "but", "that",
         "may", "not", "be", "allowed", "."],
    ]

    X_list, Y_list = corpus_to_sequences(sample_corpus, _char2idx, _label2idx)
    X_padded, Y_padded = pad_sequences_custom(X_list, Y_list)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_padded, Y_padded, test_size=0.2, random_state=42
    )

    # Example hyperparameters
    VOCAB_SIZE = len(_char2idx)  # number of distinct characters, including [PAD], [UNK], etc.
    EMBED_DIM = 64  # dimensionality of the character embedding
    LSTM_UNITS = 128  # number of units in the LSTM (per direction)
    NUM_CLASSES = 4  # for B, I, E, S labels

