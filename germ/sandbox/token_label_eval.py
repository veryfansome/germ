import json


if __name__ == '__main__':
    #lemmas = {}
    #for morphemes_file in [
    #    "data/morphemes_a_menadione.json",
    #    "data/morphemes_menage_ungainliness.json",
    #    "data/morphemes_ungainly_zyopterus.json",
    #]:
    #    with open(morphemes_file) as fd:
    #        for item in json.load(fd):
    #            lemmas[item["word"]] = {"morphemes": item["morphemes"]}
    #with open("data/morphemes.json", "w") as fd:
    #    json.dump(lemmas, fd, indent=2)
    with open("data/morphemes.json") as fd:
        lemmas = json.load(fd)

    for _ in range(2):  # Make two passes, a few will be missed in the first pass
        for word, labels in lemmas.items():
            new_morphemes = []
            for morpheme in labels["morphemes"]:
                if morpheme in lemmas and len(lemmas[morpheme]["morphemes"]) > 1:
                    new_morphemes.extend(lemmas[morpheme]["morphemes"])
                    continue
                new_morphemes.append(morpheme)
            lemmas[word] = {"morphemes": new_morphemes}

    morphemes = set()
    for word, labels in lemmas.items():
        new_morphemes = [m.strip("-") for m in labels["morphemes"]]
        print(word, new_morphemes)
        morphemes.update(new_morphemes)
    print(len(morphemes))
