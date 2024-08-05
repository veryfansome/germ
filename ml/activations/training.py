from api.models import ChatMessage, ChatRequest
from chat.openai_handlers import messages_to_transcript
from ml.bert_classifier import BertClassificationPredictor
import random


class ActivationTrainingExample:
    def __init__(self, labels: dict[str, str], messages: list[ChatMessage]):
        self.labels: dict[str, str] = labels
        self.messages: list[ChatMessage] = messages
        self.transcript_text: str = messages_to_transcript(self.to_chat_request())

    def to_chat_request(self) -> ChatRequest:
        return ChatRequest(messages=self.messages)


class ActivationTrainer:
    def __init__(self,
                 examples: list[ActivationTrainingExample],
                 trainees: dict[str, BertClassificationPredictor],
                 rounds: int):
        self.examples: list[ActivationTrainingExample] = examples
        self.trainees: dict[str, BertClassificationPredictor] = trainees
        self.rounds: int = rounds

    def train(self):
        for i in range(self.rounds):
            random.shuffle(self.examples)
            for exp in self.examples:
                for image_model_name in exp.labels.keys():
                    transcript_embeddings = self.trainees[image_model_name].generate_embeddings(exp.transcript_text)
                    self.trainees[image_model_name].train_bert_classifier(
                        exp.labels[image_model_name], transcript_embeddings
                    )
                    self.trainees[image_model_name].save()

