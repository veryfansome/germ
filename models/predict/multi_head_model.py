from transformers import DebertaV2Config, DebertaV2Model, DebertaV2PreTrainedModel
import torch.nn as nn


class MultiHeadModelConfig(DebertaV2Config):
    def __init__(self, label_maps=None, num_labels_dict=None, **kwargs):
        super().__init__(**kwargs)
        self.label_maps = label_maps or {}
        self.num_labels_dict = num_labels_dict or {}

    def to_dict(self):
        output = super().to_dict()
        output["label_maps"] = self.label_maps
        output["num_labels_dict"] = self.num_labels_dict
        return output


class MultiHeadModel(DebertaV2PreTrainedModel):
    def __init__(self, config: MultiHeadModelConfig):
        super().__init__(config)

        self.deberta = DebertaV2Model(config)
        self.classifiers = nn.ModuleDict()

        hidden_size = config.hidden_size
        for label_name, n_labels in config.num_labels_dict.items():
            self.classifiers[label_name] = nn.Sequential(
                nn.Dropout(
                    0.2  # Try 0.2 or 0.3 to see if overfitting reduces, if dataset is small or has noisy labels
                ),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, n_labels)
            )

        # Initialize newly added weights
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            **kwargs
    ):
        """
        labels_dict: a dict of { label_name: (batch_size, seq_len) } with label ids.
                     If provided, we compute and return the sum of CE losses.
        """
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )

        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        logits_dict = {}
        for label_name, classifier in self.classifiers.items():
            logits_dict[label_name] = classifier(sequence_output)
        return logits_dict
