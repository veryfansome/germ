from openai import OpenAI
from observability.logging import logging
from typing_extensions import Literal


logger = logging.getLogger(__name__)

DEFAULT_CHAT_MODEL = "gpt-4o"
DEFAULT_IMAGE_MODEL = "dall-e-3"
DEFAULT_TEMPERATURE = 0.0
ENABLED_MODELS = (
    'dall-e-3',
    #'gpt-3.5-turbo',
    'gpt-4o',
)
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


class ChatClient:

    def __init__(self, model: str = DEFAULT_CHAT_MODEL):
        self.model = model

    def complete_chat(self, chat_frame: list[dict],
                      system_message: str = None,
                      temperature: float = DEFAULT_TEMPERATURE,
                      tools: dict[str, dict] = None,
                      tool_choice: Literal['auto', 'none'] = 'none'):
        with OpenAI() as client:
            return client.chat.completions.create(
                messages=(([{"role": "system", "content": system_message}] if system_message else [])
                          + [x for x in chat_frame]),
                model=self.model,
                n=1,
                temperature=temperature,
                # A list of tools the model may call. Currently, only functions are supported as a tool. Use this to
                # provide a list of functions the model may generate JSON inputs for. A max of 128 functions are
                # supported.
                tools=tool_selection_wrapper(tools if tools else ENABLED_TOOLS),
                # Controls which (if any) tool is called by the model. none means the model will not call any tool and
                # instead generates a message. auto means the model can pick between generating a message or calling one
                # or more tools. required means the model must call one or more tools. Specifying a particular tool via
                # {"type": "function", "function": {"name": "my_function"}} forces the model to call that tool. `none`
                # is the default when no tools are present. `auto` is the default if tools are present.
                tool_choice=tool_choice)


def do_on_text(directive: str,
               text: str,
               model=DEFAULT_CHAT_MODEL,
               max_tokens=480,
               stop=None,
               temperature=DEFAULT_TEMPERATURE) -> str:
    response = OpenAI().chat.completions.create(
        messages=[{"role": "user", "content": f'{directive}: {text}'}],
        model=model,
        max_tokens=max_tokens, n=1, stop=stop, temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def generate_image(prompt: str,
                   model=DEFAULT_IMAGE_MODEL,
                   size: Literal['1024x1024', '1024x1792', '1792x1024'] = '1024x1024',
                   style: Literal['natural', 'vivid'] = 'natural') -> object:
    response = OpenAI().images.generate(
        prompt=prompt,
        model=model,
        n=1, size=size, style=style
    )
    return response.data[0].url


def tool_selection_wrapper(tools_with_callbacks: dict[str, dict]) -> list:
    # Callbacks need to be removed because OpenAI rejects them
    tools_without_callbacks = []
    for tool_spec in tools_with_callbacks.values():
        new_spec = tool_spec.copy()
        new_spec.pop("callback")
        tools_without_callbacks.append(new_spec)
    return tools_without_callbacks


def tool_wrapper(tool_func: str, arguments: dict) -> str:
    logging.info("tool call: %s(%s)", tool_func, arguments)
    if "callback" in ENABLED_TOOLS[tool_func]:
        return ENABLED_TOOLS[tool_func]["callback"](
            globals()[tool_func](**arguments),
            arguments
        )
    else:
        return globals()[tool_func](**arguments)


CHAT_COMPLETION_FUNCTIONS = {
    "dall-e-3": lambda messages: {
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": generate_image(do_on_text(
                    "Rephrase the following message as an image prompt", messages[-1]), model="dall-e-3"),
                "role": "assistant",
                "tool_calls": None,
            },
        }],
        "model": "dall-e-3",
    },
    "gpt-3.5-turbo": ChatClient(model="gpt-3.5-turbo").complete_chat,
    "gpt-4o": ChatClient(model="gpt-4o").complete_chat,
}
