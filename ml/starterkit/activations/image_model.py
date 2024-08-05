from api.models import ChatMessage
from ml.activations.training import ActivationTrainingExample

EXAMPLES = [
    ActivationTrainingExample(labels={"dall-e-2": "off", "dall-e-3": "off"}, messages=[
        ChatMessage(role="Hi", conent=""),
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
            image_model_name, model_dir="")
    trainer = ActivationTrainer(EXAMPLES, activators, germ_settings.IMAGE_MODEL_STARTERKIT_TRAINING_ROUNDS)
    trainer.train()
