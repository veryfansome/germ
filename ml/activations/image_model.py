from openai import OpenAI
import json
import logging

from api.models import ChatRequest
from bot.websocket import WebSocketEventHandler, WebSocketSender
from chat.openai_handlers import messages_to_transcript
from db.models import ImageModel, ImageModelChatRequestLink, SessionLocal
from ml.bert_classifier import new_activation_predictor
from observability.annotations import measure_exec_seconds
from settings.openai_settings import (DEFAULT_CHAT_MODEL,
                                      ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE)

logger = logging.getLogger(__name__)


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
        parameter_properties = {}
        for model_name in ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE:
            parameter_properties[model_name] = {
                "type": "boolean",
                "description": " ".join((
                    "True, if the next assistant message should be an image",
                    f"and {model_name} is the best fit based on all specified criteria,",
                    "else False."
                ))
            }
        with OpenAI() as client:
            completion = client.chat.completions.create(
                messages=([{
                    "role": "system",
                    "content": " ".join((
                        "### Best-fit Image Model Criteria\n\n",
                        "In descending order of priority, where `1.` is the most important and `4.` is the least.",
                        "1. Model has been specifically requested by the user.\n",
                        "2. Model can handle the complexity of the desired subject matter.\n",
                        "3. Model will likely be the cheapest to use.\n\n",
                        "4. Model will likely be the fastest to return an image.\n\n",
                        " ".join((
                            "When determining how complex and image should be,",
                            "assume the lowest complexity should be used",
                            "unless the user has specifically requested a more complex image.\n\n"
                        )),
                    ))
                }] + [message.dict() for message in chat_request.messages]),

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
                            "Activates the best-fit image creation model",
                            "if the next assistant message should be an image."
                        )),
                        "parameters": {
                            "type": "object",
                            "properties": parameter_properties,
                            "required": ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE
                        },
                    }
                }],
            )
            function_arguments = json.loads(
                completion.choices[0].message.tool_calls[0].function.arguments)
        logger.info(function_arguments)

        # DB updates
        bulk_insert_mapping = []
        labels = {
            "on": [],
            "off": [],
        }
        for image_model_name, activate_true in function_arguments.items():
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

        # Classifier updates
        transcript = messages_to_transcript(chat_request)
        logger.info("transcript: %s", transcript)
        for label in labels.keys():
            for image_model_name in labels[label]:
                activator = new_activation_predictor(image_model_name)  # Loads from disk on init
                transcript_embeddings = activator.generate_embeddings(transcript)
                activator.train_bert_classifier(label, transcript_embeddings)
                activator.save()