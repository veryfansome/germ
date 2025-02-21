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
            self.classifiers[label_name] = nn.Linear(hidden_size, n_labels)

        # Initialize newly added weights
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels_dict=None,
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

        total_loss = None
        loss_dict = {}
        if labels_dict is not None:
            # We'll sum the losses from each head
            loss_fct = nn.CrossEntropyLoss()
            total_loss = 0.0

            for label_name, logits in logits_dict.items():
                if label_name not in labels_dict:
                    continue
                label_ids = labels_dict[label_name]

                # A typical approach for token classification:
                # We ignore positions where label_ids == -100
                active_loss = label_ids != -100  # shape (bs, seq_len)

                # flatten everything
                active_logits = logits.view(-1, logits.shape[-1])[active_loss.view(-1)]
                active_labels = label_ids.view(-1)[active_loss.view(-1)]

                loss = loss_fct(active_logits, active_labels)
                loss_dict[label_name] = loss.item()
                total_loss += loss

        if labels_dict is not None:
            # return (loss, predictions)
            return total_loss, logits_dict
        else:
            # just return predictions
            return logits_dict
