import re


def findall_words(sentence: str) -> list[str]:
    return re.findall(r'\w+', sentence)


if __name__ == '__main__':
    print(findall_words("Hello, world! This is a test."))
