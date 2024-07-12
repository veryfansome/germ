from transformers import BertModel, BertTokenizer
# from transformers import DistilBertModel, DistilBertTokenizer
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from bot.openai_utils import ENABLED_MODELS
from observability.logging import logging

logger = logging.getLogger(__name__)


BERT_MODEL_NAME = os.getenv('BERT_MODEL_NAME', 'bert-base-uncased')
ENABLED_TOOLS = {
    "train_model_selection_neural_network": {
        "type": "function",
        "function": {
            "name": "train_model_selection_neural_network",
            "description": "Improve the model that predicts which LLM is best for replying to the user. "
                           #+ "This tool should be used when the user gives negative feedback on a previous reply.",
                           + "This tool should be used when the user gives feedback on a previous reply.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_message_that_resulted_incorrect_model_selection": {
                        "type": "string",
                        "description": "The user message that the assistant replied to, "
                                       + "that the user is giving feedback on.",
                    },
                    "correct_model": {
                        "type": "string",
                        #"description": "The name of the correct model that should have been used.",
                        "description": "The name of the correct model.",
                    }
                },
                "required": [
                    "user_message_that_resulted_incorrect_model_selection",
                    "correct_model"
                ]
            },
        },
        "callback": lambda func_args: train_model_selection_neural_network(
            func_args['user_message_that_resulted_incorrect_model_selection'],
            func_args['correct_model'])
    }
}

model_dir = os.getenv("MODEL_DIR", "/src/data/germ")
save_file = f"{model_dir}/model_selector.pth"


# Define a simple neural network for classification
class ModelSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ModelSelector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


# Load pre-trained BERT model and tokenizer
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)
bert_model_embedding_size = bert_model.config.hidden_size
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# Initialize the model, loss function, and optimizer
model_selector = ModelSelector(
    bert_model_embedding_size,  # input_dim
    128,  # hidden_dim
    len(ENABLED_MODELS)  # output_dim
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_selector.parameters(), lr=0.001)


# Function to generate embeddings
def generate_embeddings(text):
    inputs = bert_tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token
    return embeddings


# Function to load the model and optimizer state
def load_model_selector():
    if os.path.exists(save_file):
        checkpoint = torch.load(save_file)
        model_selector.load_state_dict(checkpoint['model_selector_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model_selector.eval()  # Set the model to evaluation mode


# Function to predict the best model
def predict_model(embeddings):
    load_model_selector()
    model_selector.eval()
    with torch.no_grad():
        outputs = model_selector(embeddings)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()


# Function to save the model and optimizer state
def save_model_selector():
    torch.save({
        'model_selector_state_dict': model_selector.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_file)


def train_model_selector(embeddings, label):
    model_selector.train()
    optimizer.zero_grad()
    outputs = model_selector(embeddings)
    loss = criterion(outputs, torch.tensor([label]))
    loss.backward()
    optimizer.step()


# Function to train the model
def train_model_selection_neural_network(message: str, correct_model: str):
    time_started = time.time()
    load_model_selector()
    logger.info(f"finished loading model_selector after {time.time() - time_started}s")

    train_model_selector(generate_embeddings(message), ENABLED_MODELS.index(correct_model))
    logger.info(f"finished training run after {time.time() - time_started}s")

    save_model_selector()
    logger.info(f"finished saving model_selector after {time.time() - time_started}s")

    return (f"Ok. I've updated my model selection behavior based on "
            + f"\"{message}\" and your feedback that `{correct_model}` is the correct model")


if __name__ == '__main__':
    from observability.logging import logging, setup_logging
    import pandas as pd

    setup_logging()

    logger.info("initializing model_selector")
    time_init_started = time.time()
    try:
        if not os.path.exists(save_file):
            logger.info(f"did not find {save_file}, bootstrapping")

            # Initial training data to bootstrap a viable network
            examples_dir = f"{model_dir}/examples"
            training_csv = f"{examples_dir}/prompts.csv"

            df = pd.read_csv(training_csv, delimiter=',', quotechar='"')
            df_shuffled = df.sample(frac=1).reset_index(drop=True)

            for index, row in df_shuffled.iterrows():
                time_case_started = time.time()
                train_model_selector(generate_embeddings(row['Prompt']), ENABLED_MODELS.index(row['Model']))
                logger.info(f"finished case #{index} after {time.time() - time_case_started}s")
            save_model_selector()
        logger.info(f"finished model_selector initialization after {time.time() - time_init_started}s")
    except Exception as exc:
        logger.info(f"failed to initialize model_selector after {time.time() - time_init_started}s: %s", exc)
