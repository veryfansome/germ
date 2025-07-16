import json


if __name__ == '__main__':
    lemmas = {}
    with open("data/morphemes_a_menadione.json") as fd:
        for item in  json.load(fd):
            lemmas[item["word"]] = {"morphemes": item["morphemes"]}

    cnt = 0
    for word, labels in lemmas.items():
        new_morphemes = []
        for morpheme in labels["morphemes"]:
            if morpheme in lemmas and len(lemmas[morpheme]["morphemes"]) > 1:
                print(morpheme, lemmas[morpheme]["morphemes"])
                cnt += 1
    print(cnt)
