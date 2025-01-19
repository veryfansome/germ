from tensorflow.keras import layers, models
import logging
import numpy as np
import re

logger = logging.getLogger(__name__)

# NOTE: Doesn't handle sentence direction changes where the hyphen is next to a word without a space.
naive_punctuation_pattern = re.compile(
    # Start, any start wrapper, not space
    r"((?<=^)[`'\"<\[({](?=\S)"
    # Space, any start wrapper, not space
    r"|(?<=\s)[`'\"<\[({](?=\S)"
    # Not space, any end wrapper, space|any_end_wrapper|any_end_punc|end
    r"|(?<=\S)[`'\">\])}](?=\s|[`'\">\])}]|[,:;.!?]|$)"
    # Not space, [,:;], space|end
    r"|(?<=\S)[,:;](?=\s|$)"
    # Not space, [.!?]+, space|end
    r"|(?<=\S)[.!?]+(?=\s|$))")


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
    extra_chars = ["…", "–", "—", "“", "”", "‘", "’", "¶", "§", "€", "£", "¥", "•", "±", "×", "÷", "°", "©", "®"]
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


def chunk_text(text, max_len=140):
    """
    Splits the input text into chunks of up to max_len characters, cutting at the last whitespace before max_len
    if possible. Returns a list of text chunks.

    `max_len` is 140 because that was the limit of tweets. It is long enough for tokenizing most english code and texts.
    Minimizing chunk size reduces memory requirements and, hopefully, improves inference times.
    """

    chunks = []
    start_idx = 0
    n = len(text)

    while start_idx < n:
        # If remaining text length is within max_len, take it all
        if (n - start_idx) <= max_len:
            chunks.append(text[start_idx:])
            break

        # Otherwise, look at next potential chunk from start_idx to start_idx+max_len
        candidate_end = start_idx + max_len

        # Attempt to find the last whitespace in the candidate region
        # i.e., from 'start_idx' to 'candidate_end'
        chunked_text = text[start_idx:candidate_end]
        # The last whitespace in this chunk, if any
        last_space = chunked_text.rfind(' ')

        if last_space == -1:
            # No whitespace found within chunk ->
            # we might just cut exactly at 140, if that is acceptable
            # (Alternatively, you could scan backwards for punctuation.)
            chunk = chunked_text
            next_start = candidate_end
        else:
            # last_space is an index within chunk_text
            chunk = chunked_text[:last_space]
            next_start = start_idx + last_space + 1  # +1 to move beyond the whitespace

        # Add the chunk to the list
        chunks.append(chunk)

        # Update start_idx
        start_idx = next_start

    return chunks


def compute_character_accuracy(y_true, y_pred, pad_label=0):
    """
    Computes accuracy at the character (time-step) level,
    ignoring positions with the pad_label.

    y_true, y_pred: integer arrays of shape (num_samples, seq_len)
    pad_label: the label ID used for padding. If you're using 0 for 'B' or something else,
               adjust accordingly. Some folks use -1 for pad.
    """
    total = 0
    correct = 0

    for true_seq, pred_seq in zip(y_true, y_pred):
        for t_label, p_label in zip(true_seq, pred_seq):
            if t_label == pad_label:
                # ignore padded positions
                continue
            if t_label == p_label:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


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


def decode_bies_sequence(labels, idx2label, pad_label_idx=0):
    """
    Given a sequence of label IDs (e.g., [0,1,1,2,3...])
    decode them into tokens based on B/I/E/S.
    Return a list of token boundaries as (start_index, end_index).
    """
    tokens = []
    start = -1
    for label_idx, label_id in enumerate(labels):
        # ignore pad
        if label_id == pad_label_idx:
            continue

        label = idx2label[label_id]
        if label == "B":
            start = label_idx
        elif label == "I":
            # do nothing, we remain in the middle
            pass
        elif label == "E":
            # end the token
            if start == -1:
                # invalid sequence, but handle gracefully
                start = label_idx
            tokens.append((start, label_idx))  # inclusive end
            start = -1
        elif label == "S":
            # single char token
            tokens.append((label_idx, label_idx))
            start = -1
        else:
            # unknown label?
            pass
    return tokens


def encode_char(ch, char2idx):
    """
    Convert a single character into its integer ID.
    If it's not found in char2idx, return the index for '[UNK]'.
    """
    if ch in char2idx:
        return char2idx[ch]
    else:
        return char2idx["[UNK]"]


def fit_and_test(model, corpus_kit, batch_size: int = 32, epochs: int = 10):
    model.fit(
        corpus_kit["train"]["x"], corpus_kit["train"]["y"],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(corpus_kit["val"]["x"], corpus_kit["val"]["y"]),
    )

    test_loss, test_accuracy = model.evaluate(
        corpus_kit["test"]["x"], corpus_kit["test"]["y"], batch_size=32)
    logger.info(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

    predictions = model.predict(corpus_kit["test"]["x"])  # (num_samples, seq_len, 4)
    predicted_labels = np.argmax(predictions, axis=-1)  # (num_samples, seq_len)

    char_acc = compute_character_accuracy(
        corpus_kit["test"]["y"], predicted_labels, pad_label=0)
    logger.info(f"Character-level accuracy (ignoring pad): {char_acc}")

    p, r, f1 = token_level_f1_score(
        corpus_kit["test"]["y"], predicted_labels, _idx2label, pad_label_idx=0)
    logger.info(f"Token-level F1 = {f1:.4f} (precision={p:.4f}, recall={r:.4f})")


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
    for idx, (x_seq, y_seq) in enumerate(zip(x_list, y_list)):
        seq_len = len(x_seq)
        x_padded[idx, :seq_len] = x_seq
        y_padded[idx, :seq_len] = y_seq

    return x_padded, y_padded


def naive_tokenization(text):
    tokens = []
    for blob in text.split():
        if not blob:
            continue

        matcher = naive_punctuation_pattern.search(blob)
        while matcher:
            start_idx = matcher.start()
            end_idx = matcher.end()
            if start_idx == 0:
                tokens.append(blob[start_idx:end_idx])
                blob = blob[end_idx:]
            else:
                tokens.append(blob[:start_idx])
                tokens.append(blob[start_idx:end_idx])
                blob = blob[end_idx:]
            matcher = naive_punctuation_pattern.search(blob)
        if blob:
            tokens.append(blob)
    return tokens


def process_corpus(corpus: list[list[str]], max_len: int,
                   char2idx: dict[str, int], idx2char: dict[int, str],
                   label2idx: dict[str, int], idx2label: dict[int, str],
                   name="corpus"):
    x_corpus_list, y_corpus_list = corpus_to_sequences(corpus, char2idx, label2idx)
    for i, (x_ids, y_ids) in enumerate(zip(x_corpus_list[:10], y_corpus_list[:10])):
        logger.info(f"Example {i}:\n x_ids: {[idx2char[xi] for xi in x_ids]}\n y_ids: {[idx2label[yi] for yi in y_ids]}")

    x_corpus_padded, y_corpus_padded = pad_sequences_custom(x_corpus_list, y_corpus_list, max_seq_len=max_len)
    logger.info(f"x_{name}_padded.shape {x_corpus_padded.shape}, y_{name}_padded.shape: {y_corpus_padded.shape}")
    logger.info(f"x_{name}_padded[0]: {x_corpus_padded[0]}\ny_{name}_padded[0]: {y_corpus_padded[0]}")

    x_corpus_train, x_corpus_test, y_corpus_train, y_corpus_test = train_test_split(
        x_corpus_padded, y_corpus_padded, test_size=0.2, random_state=42
    )
    # Split validation set from training set.
    x_corpus_train, x_corpus_val, y_corpus_train, y_corpus_val = train_test_split(
        x_corpus_train, y_corpus_train, test_size=0.2, random_state=42
    )
    return {
        "train": {"x": x_corpus_train, "y": y_corpus_train},
        "test": {"x": x_corpus_test, "y": y_corpus_test},
        "val": {"x": x_corpus_val, "y": y_corpus_val}}


def token_level_f1_score(y_true, y_pred, idx2label, pad_label_idx=0):
    """
    Compute approximate token-level F1 by converting each
    sequence of B/I/E/S labels to sets of token spans, then
    comparing predicted vs. gold.
    """
    tp = 0
    fp = 0
    fn = 0

    for true_seq, pred_seq in zip(y_true, y_pred):
        # decode spans from B/I/E/S
        gold_spans = set(decode_bies_sequence(true_seq, idx2label, pad_label_idx))
        pred_spans = set(decode_bies_sequence(pred_seq, idx2label, pad_label_idx))

        # overlaps for true positive
        tp += len(gold_spans.intersection(pred_spans))
        fp += len(pred_spans.difference(gold_spans))
        fn += len(gold_spans.difference(pred_spans))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


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
            for idx, c in enumerate(token):
                if idx == 0:
                    chars.append(c)
                    labels.append("B")
                elif idx == token_len - 1:
                    chars.append(c)
                    labels.append("E")
                else:
                    chars.append(c)
                    labels.append("I")
    return chars, labels


if __name__ == "__main__":
    from datetime import datetime
    from nltk.corpus import words
    from sklearn.model_selection import train_test_split
    import os
    import tensorflow.keras as keras

    from bot.lang.dependencies import words
    from observability.logging import setup_logging
    setup_logging()

    _char2idx, _idx2char = build_char_vocab()

    _label2idx = {"B": 0, "I": 1, "E": 2, "S": 3}
    _idx2label = {v: k for k, v in _label2idx.items()}

    words_corpus = [[w] for w in words.words()]
    logger.info(f"english_words_corpus[:10]: {words_corpus[:10]}")

    sample_corpus = [
        ["Hello", ",", "world", "!"],
        ["I", "love", "Python", "3.12"],
        ["You", "can't", "use", "that", "name", "because", ",", "you", "can", "see", "their", "©", "next", "to",
         "it", "."],
        ["You", "can", "email", "me", "at", "john@example.com", "."],
        ["You", "can", "check", "what's", "in", "your", "home", "directory", "by", "running", "`", "ls", "-la", "~/",
         "`", "."],
        ["Use", "a", "subnet", "of", "255.255.255.0", "or", "/24", "."],
        ["My", "router", "is", "located", "at", "192.168.1.1", "with", "a", "subnet", "of", "/23", "."],
        ["At", "my", "company", ",", "our", "internal", "network", "uses", "10.0.0.0/8"],
        ["An", "IP", "address", "might", "look", "like", "this", ":", "192.168.1.10", "."],
        ["Oh", ",", "0.0.0.0/0", "is", "special", "."],
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
        ["A", "/30", "subnet", ",", "such", "as", "10.0.0.0/30", ",", "provides", "just", "four", "addresses", "(",
         "10.0.0.0", "to", "10.0.0.3", ")", "."],
        ["It", "allows", "for", "(", "2", "^", "{", "24", "}", "-", "2", ",", "or", "16,777,214", "usable",
         "addresses", "."],
        ["First", ",", "you", "need", "to", "`", "terraform", "init", "`", "."],
        ["To", "become", "the", "root", "user", ",", "you", "can", "run", "`", "sudo", "-i", "`", ",", "but", "that",
         "may", "not", "be", "allowed", "."],
        ["To", "quickly", "see", "which", "ports", "are", "up", ",", "you", "can", "use", "`", "netstat", "-plnt",
         "`", "."],
    ]

    # Chunking texts longer than 140 characters.
    sample_long_text = (
        "This is a slightly longer example text that exceeds one hundred and forty "
        "characters. We'll chunk it by finding the last whitespace before the 140th character, "
        "and continue until the entire string is exhausted!"
    )
    sample_long_text_chunks = chunk_text(sample_long_text)
    sample_corpus += [naive_tokenization(chunk) for chunk in sample_long_text_chunks]
    logger.info(f"sample_long_text_chunks: {sample_long_text_chunks}")

    # Naive tokenization
    sample_naive_tokenization = [
        ("You don't have to `cat '<file>' | grep '<pattern>'`, you can just "
         "`grep '<pattern>' '<file>'` - drives me nuts..."),
        "Don't `sudo rm -rf /` on your machine.",
    ]
    for example in sample_naive_tokenization:
        naive_tokens = naive_tokenization(example)
        logger.info(f"naive_tokens: {naive_tokens}")
        sample_corpus.append(naive_tokens)

    ##
    # Prepare training data

    VOCAB_SIZE = len(_char2idx)  # number of distinct characters, including [PAD], [UNK], etc.
    EMBED_DIM = 64  # dimensionality of the character embedding
    LSTM_UNITS = 128  # number of units in the LSTM (per direction)
    NUM_CLASSES = 4  # for B, I, E, S labels
    MAX_SEQ_LEN = 140  # Same as max_len parameter of chunk_text_by_whitespace()

    # Word list data set

    word_corpus_kit = process_corpus(
        words_corpus, MAX_SEQ_LEN, _char2idx, _idx2char, _label2idx, _idx2label,
        name="word_corpus")

    # Sample corpus data set

    sample_corpus_kit = process_corpus(
        sample_corpus, MAX_SEQ_LEN, _char2idx, _idx2char, _label2idx, _idx2label,
        name="sample_corpus")

    ##
    # Prepare model

    model_dir = "/src/models/germ/tokenizer"
    model_files = [f for f in os.listdir(model_dir)]

    if not model_files:
        # 1) Define Inputs
        # Note: shape=(MAX_SEQ_LEN,) for fixed chunk length
        inputs = layers.Input(shape=(MAX_SEQ_LEN,), name='char_input')

        # 2) Embedding layer
        #    mask_zero=True tells Keras to ignore (mask out) padded indices (usually index 0)
        embedding = layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBED_DIM,
            name='char_embedding'
        )(inputs)

        # 3) Bidirectional LSTM
        #    return_sequences=True because we need an output vector for each time step
        bi_lstm = layers.Bidirectional(
            layers.LSTM(LSTM_UNITS, return_sequences=True),
            name='bi_lstm'
        )(embedding)

        # 4) Time-distributed output layer
        #    We apply a Dense layer with softmax to each time step independently
        outputs = layers.TimeDistributed(
            layers.Dense(NUM_CLASSES, activation='softmax'),
            name='output_softmax'
        )(bi_lstm)

        # 5) Build the model
        tokenizer_model = models.Model(inputs=inputs, outputs=outputs)

        # 6) Compile the model
        #    - 'categorical_crossentropy' if your Y is one-hot (shape=[batch, seq_len, NUM_CLASSES])
        #    - 'sparse_categorical_crossentropy' if your Y is integer-encoded (shape=[batch, seq_len])
        tokenizer_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        model_files.sort(reverse=True)  # Descending
        tokenizer_model = keras.models.load_model(f"{model_dir}/{model_files[0]}")

    # 7) Inspect the model
    tokenizer_model.summary()
    model_files.sort()

    ##
    # Train the model
    fit_and_test(tokenizer_model, word_corpus_kit)
    tokenizer_model.save(f"{model_dir}/{datetime.now().strftime("%Y%m%d%H%M%S")}.keras")
    fit_and_test(tokenizer_model, sample_corpus_kit)
