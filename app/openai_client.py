from openai import OpenAI

openai_client = OpenAI()


def get_openai_chat_response(messages) -> str:
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return completion.choices[0].message.content.strip()
