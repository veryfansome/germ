from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion , Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from starlette.concurrency import run_in_threadpool
import json
import time

from api.models import ChatRequest, ChatResponse
from bot.websocket import WebSocketEventHandler, WebSocketSender
from ml.bert_classifier import BertClassificationPredictor, new_activation_predictor
from settings.openai_settings import (DEFAULT_CHAT_MODEL,
                                      ENABLED_CHAT_MODELS, ENABLED_IMAGE_MODELS,
                                      ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE)

ACTIVATION_PREDICTORS: dict[str, BertClassificationPredictor] = {}
CHAT_HANDLERS: dict[str, WebSocketEventHandler] = {}


class ChatModelEventHandler(WebSocketEventHandler):
    def __init__(self, model: str):
        self.model = model

    async def on_receive(self,
                         chat_session_id: int, chat_request_received_id: int,
                         chat_request: ChatRequest, response_sender: WebSocketSender):
        completion = await run_in_threadpool(self.do_chat_completion, chat_request)
        await response_sender.send_chat_response(ChatResponse(response=completion))

    def do_chat_completion(self, chat_request: ChatRequest) -> ChatCompletion:
        with OpenAI() as client:
            return client.chat.completions.create(
                messages=[message.dict() for message in chat_request.messages],
                model=self.model,
                n=1,
                temperature=chat_request.temperature,
            )


class ImageModelEventHandler(WebSocketEventHandler):
    def __init__(self, model: str):
        self.model = model

    async def on_receive(self,
                         chat_session_id: int, chat_request_received_id: int,
                         chat_request: ChatRequest, response_sender: WebSocketSender):
        markdown_image = await run_in_threadpool(self.generate_markdown_image, chat_request)
        await response_sender.send_chat_response(ChatResponse(response=ChatCompletion(
            id='none',
            choices=[Choice(
                finish_reason='stop',
                index=0,
                message=ChatCompletionMessage(
                    content=markdown_image,
                    role='assistant'
                ),
            )],
            created=int(time.time()),
            object="chat.completion",
            model=self.model,
        )))

    def generate_markdown_image(self, chat_request: ChatRequest):
        image_model_inputs = generate_image_model_inputs(chat_request)
        with OpenAI() as client:
            submission = client.images.generate(
                prompt=image_model_inputs['prompt'], model=self.model, n=1,
                size=image_model_inputs['size'], style=image_model_inputs['style']
            )
            return f"[![{image_model_inputs['prompt']}]({submission.data[0].url})]({submission.data[0].url})"


def messages_to_transcript(chat_request: ChatRequest):
    """
    GPT-4o recommended the following format because it captures directionality and provides clear boundaries to
    messages, which is helpful when trying to get embeddings that capture the context of the conversation.

    [USER] Hi! [ASSISTANT]
    [ASSISTANT] Hey! How can I help you? [USER]
    [USER] Tell me a joke [ASSISTANT]

    :param chat_request:
    :return: transcript as a string
    """
    transcript_lines = []
    for message in chat_request.messages:
        end_tag = "user" if message.role == "assistant" else "assistant"
        transcript_lines.append(f"[{message.role.upper()}] {message.content} [{end_tag.upper()}]")
    return "\n".join(transcript_lines)


def generate_image_model_inputs(chat_request: ChatRequest):
    with OpenAI() as client:
        completion = client.chat.completions.create(
            messages=([{
                "role": "system",
                "content": "The user wants an image."
            }] + [message.dict() for message in chat_request.messages]),

            model=DEFAULT_CHAT_MODEL, n=1, temperature=chat_request.temperature,
            tool_choice={
                "type": "function",
                "function": {"name": "generate_image"}
            },
            tools=[{
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Return a generated image, given a text prompt, image size, and style.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The text prompt for the image model."
                            },
                            "size": {
                                "type": "string",
                                "description": "Must be one of '1024x1024', '1024x1792', or '1792x1024'."
                            },
                            "style": {
                                "type": "string",
                                "description": "Must be either 'natural' or 'vivid'."
                            },
                        },
                        "required": ["prompt", "size", "style"]
                    },
                }
            }],
        )
        return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)


for m in ENABLED_CHAT_MODELS:
    CHAT_HANDLERS[m] = ChatModelEventHandler(m)
for m in ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE:
    ACTIVATION_PREDICTORS[m] = new_activation_predictor(m)
#for m in ENABLED_IMAGE_MODELS:
#    CHAT_HANDLERS[m] = ImageModelEventHandler(m)
