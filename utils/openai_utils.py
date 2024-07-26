from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from typing_extensions import Literal
import json
import os
import tiktoken
import time

from api.models import ChatMessage
from observability.logging import logging

executor = ThreadPoolExecutor(max_workers=os.getenv("MAX_EXECUTOR_WORKERS", 1))
logger = logging.getLogger(__name__)

DEFAULT_CHAT_MODEL = "gpt-4o"
DEFAULT_IMAGE_MODEL = "dall-e-3"
DEFAULT_TEMPERATURE = 0.0
ENABLED_MODELS = (
    DEFAULT_CHAT_MODEL,
    DEFAULT_IMAGE_MODEL,
    'dall-e-2',
    'gpt-3.5-turbo',
)
ENABLED_TOKEN_LIMIT_CHECKS = {
    DEFAULT_CHAT_MODEL: True,
    DEFAULT_IMAGE_MODEL: False,
    'dall-e-2': False,
    'gpt-3.5-turbo': True,
}
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
        "callback": lambda func_args: generate_image(func_args['prompt'])
    }
}


class ChatClient:

    def __init__(self, model: str = DEFAULT_CHAT_MODEL):
        self.model = model

    def complete_chat(self, chat_frame: list[dict],
                      system_message: str = None,
                      temperature: float = DEFAULT_TEMPERATURE,
                      tools: dict[str, dict] = None,
                      tool_choice: Literal['auto', 'none'] = 'none') -> ChatCompletion:
        messages = (([{"role": "system", "content": system_message}] if system_message else [])
                    + [x for x in chat_frame])
        logger.debug("messages: %s", messages)
        with OpenAI() as client:
            return client.chat.completions.create(
                messages=messages,
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


class ImageClient:

    def __init__(self, model: str = DEFAULT_IMAGE_MODEL):
        self.model = model

    def complete_chat(self, chat_frame: list[ChatMessage],
                      system_message: str = None,
                      temperature: float = DEFAULT_TEMPERATURE,
                      **kwargs) -> ChatCompletion:
        new_chat_message: ChatMessage = chat_frame[-1]
        prompt = do_on_text("Rephrase the following message as an image prompt", new_chat_message.content,
                            system_message=system_message, temperature=temperature)
        # TODO: Use a model to predict size and style
        logger.info('image prompt: %s', prompt)
        url = self.generate_image(prompt, size='1024x1024', style='natural')
        return ChatCompletion(
            id='none',
            choices=[Choice(
                finish_reason='stop',
                index=0,
                message=ChatCompletionMessage(
                    content=f"[![{prompt}]({url})]({url})",
                    role='assistant'
                ),
            )],
            created=int(time.time()),
            object="chat.completion",
            model=self.model,
        )

    def generate_image(self, prompt: str,
                       size: Literal['1024x1024', '1024x1792', '1792x1024'] = '1024x1024',
                       style: Literal['natural', 'vivid'] = 'natural') -> object:
        with OpenAI() as client:
            response = client.images.generate(
                prompt=prompt,
                model=self.model,
                n=1, size=size, style=style
            )
            return response.data[0].url


def do_on_text(directive: str,
               text: str,
               model=DEFAULT_CHAT_MODEL,
               max_tokens=480,
               stop=None,
               system_message: str = None,
               temperature=DEFAULT_TEMPERATURE) -> str:
    with OpenAI() as client:
        response = client.chat.completions.create(
            messages=(([{"role": "system", "content": system_message}] if system_message else []) +
                      [{"role": "user", "content": f'{directive}: {text}'}])
            ,
            model=model,
            max_tokens=max_tokens, n=1, stop=stop, temperature=temperature,
        )
        return response.choices[0].message.content.strip()


def generate_image(prompt: str,
                   model=DEFAULT_IMAGE_MODEL,
                   size: Literal['1024x1024', '1024x1792', '1792x1024'] = '1024x1024',
                   style: Literal['natural', 'vivid'] = 'natural') -> object:
    url = ImageClient(model=model).generate_image(prompt, size=size, style=style)
    return f"[![{prompt}]({url})]({url})"


def handle_feedback(chat_frame: list[ChatMessage],
                    tools: dict[str, dict]) -> ChatCompletion:
    if is_feedback(chat_frame[-1].content) == 'Yes':
        with OpenAI() as client:
            completion = client.chat.completions.create(
                messages=[{
                    "role": "system",
                    "content": ' '.join((
                        "The user's last message provides feedback.",
                        "Based on this feedback, use a tool to respond if appropriate."
                        "If no tool seems appropriate, respond, 'Missing appropriate tools'."
                    ))}] + [m for m in chat_frame],  # Comprehension for list/tuple concatenation
                model=DEFAULT_CHAT_MODEL, n=1, temperature=DEFAULT_TEMPERATURE,
                tool_choice='auto', tools=tool_selection_wrapper(tools)
            )
            completion_message = handle_tool_calls_in_completion_response(completion, tools=tools)
            completion.choices[0].message = completion_message
            return completion
    else:
        return ChatCompletion(
            id='none',
            choices=[Choice(
                finish_reason='stop',
                index=0,
                message=ChatCompletionMessage(
                    content="Not feedback.",
                    role='assistant'
                ),
            )],
            created=int(time.time()),
            object="chat.completion",
            model=DEFAULT_CHAT_MODEL,
        )


# If required, call out to tools and intercept the completed message.
def handle_tool_calls_in_completion_response(completion: ChatCompletion,
                                             tools: dict[str, dict] = None) -> ChatCompletionMessage:
    completion_message = completion.choices[0].message
    logger.debug("response: %s", completion_message.content)

    # If tools should be used, call them.
    if completion_message.tool_calls:
        response_frame = [{
            "role": "system",
            "content": "Combine the assistant's messages into a single message.",
        }]
        for tool_call in completion_message.tool_calls:
            tool_args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "multi_tool_use.parallel":
                # TODO: Logging for now to see what this looks like.
                logging.info("multi_tool_use.parallel completion_message: %s", completion_message.dict())
            elif tool_call.function.name not in (tools if tools else ENABLED_TOOLS):
                raise RuntimeError(f"Unknown tool call: {tool_call.function.name}")
            elif "callback" in (tools if tools else ENABLED_TOOLS)[tool_call.function.name]:
                response_frame.append({
                    "role": "assistant",
                    "content": (tools if tools else ENABLED_TOOLS)[tool_call.function.name]["callback"](tool_args)
                })
            elif tool_call.function.name in globals():
                response_frame.append({
                    "role": "assistant",
                    "content": globals()[tool_call.function.name](**tool_args)
                })
        response_frame_size = len(response_frame)
        if response_frame_size == 2:  # One besides system message
            completion_message.content = response_frame[1]['content']
        elif response_frame_size > 2:
            with OpenAI() as client:
                summary_completion = client.chat.completions.create(
                    messages=response_frame,
                    model=DEFAULT_CHAT_MODEL,
                    n=1, temperature=DEFAULT_TEMPERATURE
                )
                completion_message.content = summary_completion.choices[0].message.content
    return completion_message


def is_feedback(message: str, system_message: str = None) -> str:
    prompt = ' '.join((
        "Is the **Message** below expressing agreement/disagreement, approval/disapproval,"
        "pointing out a correct choice, suggesting an alternate choice, or otherwise giving feedback?",

        "If no, answer 'No'.",

        "If yes, answer 'Yes', except if the message expresses agreement/approval, like \"you got it right\",",
        "or disagreement/disapproval, like \"that's not right\",",
        "but does not specifically mention a preferred choice or alternative,",
        "don't answer 'Yes, instead, answer 'Partial'.",
    ))
    completion_content = prompt + f"""
**Message**: {message}
"""
    logger.info('checking if message is user giving feedback')
    with OpenAI() as client:
        completion = client.chat.completions.create(
            messages=(([{"role": "system", "content": system_message}] if system_message else [])
                      + [{"role": "user", "content": completion_content}]),
            model=DEFAULT_CHAT_MODEL, n=1, temperature=0.0
        )
        completion_message_content = completion.choices[0].message.content.strip().strip('.')
        logger.info("is feedback answer: %s", completion_message_content)
        return completion_message_content


def is_token_limit_check_enabled(model: str) -> bool:
    return False if model not in ENABLED_TOKEN_LIMIT_CHECKS else ENABLED_TOKEN_LIMIT_CHECKS[model]


def summarize_multiple_completions(completions: list[ChatCompletion]) -> ChatCompletion:
    completions_len = len(completions)
    if completions_len == 1:
        return completions[0]
    elif completions_len > 1:
        prompt = "Summarize the following **Messages**:" + """
# **Messages**:
"""
        logger.info(f"consolidating {completions_len} completions:\n{prompt}")
        for completion in completions:
            prompt += f"\n- {completion.message.content.strip()}"
        with OpenAI() as client:
            summary_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=DEFAULT_CHAT_MODEL,
                n=1, temperature=0.0
            )
            return summary_completion


def tool_selection_wrapper(tools_with_callbacks: dict[str, dict]) -> list:
    # Callbacks need to be removed because OpenAI rejects them
    tools_without_callbacks = []
    if tools_with_callbacks:
        for tool_spec in tools_with_callbacks.values():
            new_spec = tool_spec.copy()
            if 'callback' in new_spec:
                new_spec.pop("callback")
            tools_without_callbacks.append(new_spec)
    return tools_without_callbacks


def trim_chat_frame(messages: list[ChatMessage], model, system_message: str = None):
    chat_frame = messages
    if is_token_limit_check_enabled(model):
        # Trim message list to avoid hitting selected model's token limit.
        enc = tiktoken.encoding_for_model(model)
        total_tokens = len(enc.encode(system_message))
        reversed_chat_frame = []
        for chat_message in reversed(messages):  # In reverse because message list is ordered from oldest to newest.
            message_tokens = len(enc.encode(chat_message.content))
            # If adding `message_tokens` pushes us over the limit, stop appending
            if message_tokens + total_tokens > enc.max_token_value:
                break
            total_tokens += message_tokens
            reversed_chat_frame.append(chat_message)
        chat_frame = tuple(reversed(reversed_chat_frame))  # Undo previously reversed order
    return chat_frame


CHAT_COMPLETION_FUNCTIONS = {
    DEFAULT_CHAT_MODEL: ChatClient(model=DEFAULT_CHAT_MODEL).complete_chat,
    DEFAULT_IMAGE_MODEL: ImageClient(model=DEFAULT_IMAGE_MODEL).complete_chat,
    "dall-e-2": ImageClient(model="dall-e-2").complete_chat,
    "gpt-3.5-turbo": ChatClient(model="gpt-3.5-turbo").complete_chat,
}
