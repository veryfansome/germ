from openai import OpenAI
from observability.logging import logging

logger = logging.getLogger(__name__)

DEFAULT_CHAT_MODEL = "gpt-4o"
DEFAULT_IMAGE_MODEL = "dall-e-3"
ENABLED_TOOLS = {
    "generate_image": {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image from a textual prompt. Returns the image's URL string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The textual prompt used to generate the image."
                    }
                },
                "required": ["prompt"]
            },
        },
        "callback": lambda url, arguments: f"[![{arguments['prompt']}]({url})]({url})"
    }
}


def do_on_text(directive: str, text: str, model=DEFAULT_CHAT_MODEL, max_tokens=140, stop=None, temperature=0.0) -> str:
    response = OpenAI().chat.completions.create(
        messages=[{"role": "user", "content": f'{directive}: {text}'}],
        model=model,

        max_tokens=max_tokens, n=1, stop=stop, temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def generate_image(prompt: str, model=DEFAULT_IMAGE_MODEL) -> object:
    response = OpenAI().images.generate(
        prompt=prompt,
        model=model,
        n=1,
    )
    return response.data[0].url


def tool_selection_wrapper() -> list:
    tools = []
    for tool_spec in ENABLED_TOOLS.values():
        new_spec = tool_spec.copy()
        new_spec.pop("callback")
        tools.append(new_spec)
    return tools


def tool_wrapper(tool_func: str, arguments: dict) -> str:
    logging.info("tool call: %s(%s)", tool_func, arguments)
    if "callback" in ENABLED_TOOLS[tool_func]:
        return ENABLED_TOOLS[tool_func]["callback"](
            globals()[tool_func](**arguments),
            arguments
        )
    else:
        return globals()[tool_func](**arguments)
