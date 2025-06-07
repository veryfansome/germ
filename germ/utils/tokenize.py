from germ.utils.patterns import naive_tokenize_pattern


def naive_tokenize(text: str):
    return [t for t in naive_tokenize_pattern.split(text)
            if t != ""
            and not t.startswith(" ")
            and not t.startswith("\t")]


if __name__ == "__main__":
    from germ.observability.logging import setup_logging
    setup_logging()

    print(naive_tokenize("Hello  world!"))
    print(naive_tokenize("Hello?"))
    print(naive_tokenize("Hello, I'm C-3PO. Human-cyborg relations."))
    print(naive_tokenize("So then I said, \"Hello world!\""))
    print(naive_tokenize("'''' \"\"\"\""))
    print(naive_tokenize("He'll have to carry that?!"))
    print(naive_tokenize("Yes, that makes -what are you doing?"))
    print(naive_tokenize("Sure, I can- did you see that??"))
    print(naive_tokenize("I don't think I have a short attention—squirrel!!"))
    print(naive_tokenize("The localhost address is 127.0.0.1."))
    print(naive_tokenize("\t- This is a Foo."))
    print(naive_tokenize("This is a thing -- this is another thing."))
