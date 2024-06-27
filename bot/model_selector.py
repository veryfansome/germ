import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

from bot.openai_utils import ENABLED_MODELS

BERT_MODEL_NAME = 'bert-base-uncased'  # TODO: Picked by gpt-4o, may not be the most current.
ENABLED_TOOLS = {
    "train_model_selection_neural_network": {
        "type": "function",
        "function": {
            "name": "train_model_selection_neural_network",
            "description": "Improve the model that predicts which LLM is best for replying to the user. "
                           + "This tool should be used when the user gives negative feedback on a previous reply.",
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
                        "description": "The name of the correct model that should have been used."
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
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# Initialize the model, loss function, and optimizer
model_selector = ModelSelector(
    768,  # input_dim, # BERT embedding size
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
def load_model_selector(file_path):
    checkpoint = torch.load(file_path)
    model_selector.load_state_dict(checkpoint['model_selector_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model_selector.eval()  # Set the model to evaluation mode


# Function to predict the best model
def predict_model(embeddings):
    model_selector.eval()
    with torch.no_grad():
        outputs = model_selector(embeddings)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()


# Function to save the model and optimizer state
def save_model_selector(file_path):
    torch.save({
        'model_selector_state_dict': model_selector.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)


def train_model_selector(embeddings, label):
    model_selector.train()
    optimizer.zero_grad()
    outputs = model_selector(embeddings)
    loss = criterion(outputs, torch.tensor([label]))
    loss.backward()
    optimizer.step()


# Function to train the model
def train_model_selection_neural_network(message: str, correct_model: str):
    train_model_selector(generate_embeddings(message), ENABLED_MODELS.index(correct_model))
    return (f"Ok. I've updated my model selection behavior based on "
            + f"\"{message}\" and your feedback that `{correct_model}` is the correct model")


if __name__ == '__main__':
    print("woot")
