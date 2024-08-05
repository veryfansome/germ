from api.models import ChatMessage
from ml.activations.training import ActivationTrainingExample

EXAMPLES = [
    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "off"}, messages=[
        ChatMessage(role="user", conent="Generate an image of a dog"),
    ]),
    ActivationTrainingExample(labels={"dall-e-2": "on", "dall-e-3": "off"}, messages=[
        ChatMessage(role="user", conent="hi"),
        ChatMessage(role="assistant", conent="Hello! How can I assist you today?"),
        ChatMessage(role="user", conent="generate an image of a house"),
    ]),
]

if __name__ == "__main__":
    from ml.activations.image_model import (ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE,
                                            new_activation_predictor)
    from ml.activations.training import ActivationTrainer
    from observability.logging import logging, setup_logging
    from settings import germ_settings

    setup_logging()
    logger = logging.getLogger(__name__)

    activators = {}
    for image_model_name in ENABLED_IMAGE_MODELS_FOR_TRAINING_DATA_CAPTURE:
        activators[image_model_name] = new_activation_predictor(
            image_model_name, model_dir=germ_settings.STARTERKIT_DIR)
    trainer = ActivationTrainer(EXAMPLES, activators, germ_settings.IMAGE_MODEL_STARTERKIT_TRAINING_ROUNDS)
    trainer.train()
