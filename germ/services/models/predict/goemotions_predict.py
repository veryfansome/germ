from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import torch

from germ.utils import get_torch_device


class GoEmotionsPredictor:
    def __init__(self, model_name_or_path: str, subfolder=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, subfolder=subfolder)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, subfolder=subfolder)

        self.label_names = getattr(self.model.config, "label_names", None)
        self.per_label_thresh = getattr(self.model.config, "per_label_thresholds", None)
        self.global_thresh = getattr(self.model.config, "best_global_threshold", 0.65)

        self.device = get_torch_device()
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts, use_per_label=True):
        """
        Args:
          texts (list[str]): A list of raw text strings to classify.
          use_per_label (bool): If True, apply per-label thresholds. If False, apply global threshold.
        Returns:
          A list of dicts, each with {"text": ..., "predicted_labels": [...]}
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        # 1) Run the model to get logits
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits  # shape: (batch_size, num_labels)
            probs = torch.sigmoid(logits).cpu().numpy()  # shape: (batch_size, num_labels)

        # 2) Determine predictions by thresholding
        if use_per_label:
            # Use per-label thresholds
            threshold_array = np.array(self.per_label_thresh)
            preds = (probs >= threshold_array).astype(int)  # shape: (batch_size, num_labels)
        else:
            # Use global threshold
            preds = (probs >= self.global_thresh).astype(int)

        # 3) Convert integer predictions to label names
        results = []
        for i, text in enumerate(texts):
            row_preds = preds[i]
            predicted_labels = [self.label_names[j] for j, val in enumerate(row_preds) if val == 1]
            results.append({"text": text, "emotions": predicted_labels})

        return results
