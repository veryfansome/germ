import requests


def get_definitions(word):
    """
    Retrieve definitions and alternative forms for a given word
    from the DictionaryAPI.dev service.
    """
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: unable to retrieve definition for '{word}'.")
        return None

    data = response.json()

    # The structure of the response can vary,
    # so let's parse it carefully.
    definitions = []
    alternative_forms = []

    # 'data' is usually a list of entries
    for entry in data:
        # 'meanings' is typically a list of meaning objects
        for meaning in entry.get("meanings", []):
            part_of_speech = meaning.get("partOfSpeech")
            for definition in meaning.get("definitions", []):
                definition_text = definition.get("definition")
                definitions.append((part_of_speech, definition_text))

        # 'word' or 'phonetics' might give additional info about alternative forms
        if "word" in entry:
            alternative_forms.append(entry["word"])

    return {
        "definitions": definitions,
        "alternative_forms": list(set(alternative_forms))  # Remove duplicates
    }


# Example usage
if __name__ == "__main__":
    test_cases = [
        "assistance",
        "help",
        #"person",
        #"create",
        #"creation",
        #"help",
        #"make",
        #"want",
    ]

    for case in test_cases:
        result = get_definitions(case)
        if result:
            print(f"Definitions for '{case}':")
            for pos, definition_text in result["definitions"]:
                print(f"• {pos}: {definition_text}")

            print("\nAlternative forms or spellings:")
            for form in result["alternative_forms"]:
                print(f"• {form}")