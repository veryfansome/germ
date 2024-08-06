import json

from api.models import ChatRequest
from ml.activations.image_model import get_activation_labels
from ml.starterkit.activations.image_model import EXAMPLES


def test_examples():
    for exp in EXAMPLES:
        fetched_labels = {
            k: "on" if v else "off" for k, v in get_activation_labels(ChatRequest(messages=exp.messages)).items()
        }
        mismatched_labels = []
        for image_model_name in fetched_labels.keys():
            if fetched_labels[image_model_name] != exp.labels[image_model_name]:
                mismatched_labels.append((image_model_name, fetched_labels, f"failed example {exp.__dict__}"))
        failed_cnt = len(mismatched_labels)
        assert failed_cnt == 0, f"found {failed_cnt} mismatches: {json.dumps(mismatched_labels, indent=4)}"
