import re

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
