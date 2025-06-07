import asyncio
from openai import AsyncOpenAI

async_openai_client = AsyncOpenAI()

async def segment_with_llm(word: str):
    response = await async_openai_client.chat.completions.create(
        messages=[
            {"role": "system",
             "content": ("You are an English linguist specializing in morphemes old and new. "
                         "When the user gives your a word, return it's free and bound morphemes in order.")},
            {"role": "user", "content": word},
        ],
        model="gpt-4o",
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
        n=1, timeout=10)
    return response.choices[0].message.content

if __name__ == "__main__":
    # Quick sanity check on the examples requested by the user
    examples = [
        "affixes",
        "antidepressant",
        "antidisestablishmentarianism",
        "candidate",
        "collaborating",
        "corruption",
        "disproportionately",
        "eruption",
        "interruption",
        "monomorphemic",
        "running",
        "uncharacteristically",
    ]
    vocab = set()
    for w in examples:
        #segs = segment(w)
        segs = asyncio.run(segment_with_llm(w))
        print(f"{w}: {segs}")
        #vocab.update(segs)
    print(f"{len(vocab)} tokens in vocabulary")
