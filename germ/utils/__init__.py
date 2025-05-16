import logging
import torch

logger = logging.getLogger(__name__)


def get_torch_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple Silicon MPS
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"using torch in {device} mode")
    return device
