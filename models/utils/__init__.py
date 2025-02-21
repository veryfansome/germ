from huggingface_hub import hf_hub_download
import itertools
import logging
import sentencepiece as spm
import torch

logger = logging.getLogger(__name__)

sp_model_path = hf_hub_download(repo_id="veryfansome/multi-classifier", filename="sp.model")
sp = spm.SentencePieceProcessor()
sp.LoadFromFile(sp_model_path)


def get_torch_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple Silicon MPS
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"using {device}")
    return device


def sp_tokenize(text: str):
    return list(itertools.chain.from_iterable([s.strip("▁").split("▁") for s in sp.EncodeAsPieces(text)]))
