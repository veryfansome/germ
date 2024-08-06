from openai import OpenAI
from starlette.concurrency import run_in_threadpool
import asyncio
import json

from api.models import ChatRequest
from bot.websocket import WebSocketEventHandler, WebSocketSender
from chat.openai_handlers import messages_to_transcript
from db.models import ImageModel, ImageModelChatRequestLink, SessionLocal
from ml.bert_classifier import new_activation_predictor
from observability.annotations import measure_exec_seconds
from observability.logging import logging
from settings.openai_settings import (DEFAULT_CHAT_MODEL,
                                      ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE)

logger = logging.getLogger(__name__)

# Maps to function.parameters.properties
OPENAI_FUNCTION_PARAMETERS_PROPERTIES = {}
for model_name in ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE:
    OPENAI_FUNCTION_PARAMETERS_PROPERTIES[model_name] = {
        "type": "boolean",
        "description": " ".join((
            "True, if the user is asking for an image",
            f"_**AND**_ `{model_name}` meets the **Activation Criteria**, else False."
        ))
    }


class ManagedImageModel:
    def __init__(self, image_model_id: int):
        self.image_model_id = image_model_id


class ImageModelActivationTrainingDataEventHandler(WebSocketEventHandler):
    def __init__(self):
        self.managed_image_models = {}
        with SessionLocal() as session:
            results = session.query(ImageModel).all()
            if not results:
                missing_names = ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE
            else:
                found_names = set([r.image_model_name for r in results])
                missing_names = list(
                    set(ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE) - found_names)
            if missing_names:
                session.bulk_insert_mappings(ImageModel, [{"image_model_name": name} for name in missing_names])
                session.commit()
                results = session.query(ImageModel).all()
            for r in results:
                self.managed_image_models[r.image_model_name] = ManagedImageModel(r.image_model_id)

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self,
                         chat_session_id: int, chat_request_received_id: int,
                         chat_request: ChatRequest, response_sender: WebSocketSender):
        """
        Use GPT to generate labels. Update chat request to label links in DB. Update classifier.

        :param chat_session_id:
        :param chat_request_received_id:
        :param chat_request:
        :param response_sender:
        :return: None
        """
        activation_labels = await run_in_threadpool(get_activation_labels, chat_request)
        logger.info(activation_labels)
        training_labels = await run_in_threadpool(self.update_image_model_chat_request_link,
                                                  chat_request_received_id, activation_labels)

        async def training_job():
            await run_in_threadpool(update_activators, chat_request, training_labels)
        await asyncio.create_task(training_job())

    def update_image_model_chat_request_link(self,
                                             chat_request_received_id: int,
                                             activation_labels: dict[str, bool]) -> dict[str, list[str]]:
        bulk_insert_mapping = []
        labels = {
            "on": [],
            "off": [],
        }
        for image_model_name, activate_true in activation_labels.items():
            if activate_true:
                bulk_insert_mapping.append({
                    "chat_request_received_id": chat_request_received_id,
                    "image_model_id": self.managed_image_models[image_model_name].image_model_id,
                })
                labels['on'].append(image_model_name)
            else:
                labels['off'].append(image_model_name)
        with SessionLocal() as session:
            session.bulk_insert_mappings(ImageModelChatRequestLink, bulk_insert_mapping)
            session.commit()
        return labels


def get_activation_labels(chat_request: ChatRequest) -> dict[str, bool]:
    with OpenAI() as client:
        completion = client.chat.completions.create(
            messages=[message.dict() for message in chat_request.messages],
            model=DEFAULT_CHAT_MODEL, n=1, temperature=chat_request.temperature,
            tool_choice={
                "type": "function",
                "function": {"name": "image_creation_activator"}
            },
            tools=[{
                "type": "function",
                "function": {
                    "name": "image_creation_activator",
                    "description": " ".join((
                        "Activates one or more image creation models to respond to the user.\n\n"
                        "### Activation Criteria\n\n",
                        "1. Model has been specifically requested by the user in the conversation.\n",
                        "OR\n",
                        "2. Model can handle the complexity of the desired subject matter.\n",
                    )),
                    "parameters": {
                        "type": "object",
                        "properties": OPENAI_FUNCTION_PARAMETERS_PROPERTIES,
                        "required": ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE
                    },
                }
            }],
        )
        return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)


def update_activators(chat_request: ChatRequest, training_labels: dict[str, list[str]]):
    transcript = messages_to_transcript(chat_request)
    logger.info("transcript: %s", transcript)
    for label in training_labels.keys():
        for image_model_name in training_labels[label]:
            activator = new_activation_predictor(image_model_name)  # Loads from disk on init
            transcript_embeddings = activator.generate_embeddings(transcript)
            activator.train_bert_classifier(label, transcript_embeddings)
            activator.save()
