import asyncio
import aiofiles
import json
import re
from nltk.corpus.reader import WordNetCorpusReader
from openai import AsyncOpenAI


async_openai_client = AsyncOpenAI()
oewn_path = "data/oewn2024"
sem = asyncio.Semaphore(30)  # Limit concurrency to 10 requests at a time


async def main(words: list[str], batch_size: int = 1000):
    """
    Processes the given words in batches, respecting the concurrency limit.
    Writes out a large JSON array incrementally so we don't keep everything in memory.
    """
    async with aiofiles.open("data/morphemes.json", "w") as f:
        # Write the start of a JSON array
        await f.write("[\n")

        first_record = True
        for i in range(0, len(words), batch_size):
            batch = words[i : i + batch_size]

            # Create tasks for this batch
            tasks = [asyncio.create_task(segment_with_llm(w)) for w in batch]

            # Gather results for all tasks in this batch
            batch_results = await asyncio.gather(*tasks)

            # Write each result as part of a JSON array
            for record in batch_results:
                if not first_record:
                    await f.write(",\n")  # separate array elements with commas
                else:
                    first_record = False
                await f.write(json.dumps(record, ensure_ascii=False, indent=2))

        # Close the JSON array
        await f.write("\n]\n")


def process_label(label_token: str) -> str:
    label_token = label_token.lower().strip()
    return label_token


async def segment_with_llm(word: str):
    processed_labels = []
    attempt_cnt = 0
    while not processed_labels and attempt_cnt <= 10:
        attempt_cnt += 1
        try:
            async with sem:
                response = await async_openai_client.chat.completions.create(
                    messages=[
                        {"role": "system",
                         "content": (
                             "You are an English morpheme specialist. "

                             "Given a word, return its free and bound morphemes in order. "

                             "Use a '-' to indicate a bound morpheme and the direction of the free morpheme being modified. "

                             "Focus on meaning: 'understand' should not be split into 'under-' and 'stand' or 'under' and "
                             "'-stand' because the meaning of the word is not derived from the meanings of those parts. "

                             "On the other hand, 'misunderstood' should be split into 'mis-' and 'understood'."
                         )},
                        {"role": "user",
                         "content": word
                         },
                    ],
                    #model="gpt-4o",
                    model="o3-mini",
                    #model="o1",
                    reasoning_effort="high",
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "morphemes",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "morphemes": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["morphemes"]
                            }
                        }
                    },
                    n=1, timeout=180)
                raw_label = json.loads(response.choices[0].message.content)
                if "morphemes" in raw_label:
                    processed_labels.extend([process_label(t) for t in raw_label["morphemes"]])
        except Exception:
            print(f"Error processing word: {word}, attempt {attempt_cnt}")
    result = {"word": word, "morphemes": processed_labels}
    print(result)
    return result


if __name__ == "__main__":
    # Quick sanity checks and custom additions
    examples = {
        "a",
        "affixes",
        "antidepressant",
        "antidisestablishmentarianism",
        "candidate",
        "collaborating",
        "corruption",
        "disproportionately",
        "eruption",
        "interruption",
        "misunderstood",
        "monomorphemic",
        "running",
        "something",
        "the",
        "uncharacteristically",
        "understand",
        "understood",
    }

    lower_alpha_pat = re.compile(r"^[a-z]+$")  # Lowercase alphabetic words

    reader = WordNetCorpusReader(oewn_path, omw_reader=None)
    for synset in reader.all_synsets():
        for lemma in synset.lemmas():
            lemma_components = lemma.name().split('_')
            examples.update([c for c in lemma_components if lower_alpha_pat.match(c) and len(c) > 1])


    examples = list(examples)
    examples.sort()
    print(len(examples))
    print(examples[66999:][:100])  # Print first 10 examples for visual inspection

    #asyncio.run(main(examples))
    #asyncio.run(main(examples[67000:]))
