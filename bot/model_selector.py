import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

from bot.openai_utils import ENABLED_MODELS

BERT_MODEL_NAME = 'bert-base-uncased'  # TODO: Picked by gpt-4o, may not be the most current.


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


# Function to train the model
def train_model(embeddings, label):
    model_selector.train()
    optimizer.zero_grad()
    outputs = model_selector(embeddings)
    loss = criterion(outputs, torch.tensor([label]))
    loss.backward()
    optimizer.step()