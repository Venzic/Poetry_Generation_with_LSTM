##### Imports #####
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

##### Data Preparation #####
# Load poetry data from poems.txt with utf-8 encoding
with open("poems.txt", "r", encoding="utf-8") as file:
    data = file.read().lower()

# Update character set based on poems.txt data
chars = sorted(set(data))
data_size, char_size = len(data), len(chars)

# Create mappings from characters to indices and vice versa
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

print(f"Data size: {data_size}, Char size: {char_size}")

# Prepare training data using character sequences
sequence_length = 10  # Use a context of 10 characters
train_X = []
train_y = []
for i in range(data_size - sequence_length):
    sequence = data[i:i + sequence_length]
    next_char = data[i + sequence_length]
    train_X.append([char_to_idx[c] for c in sequence])
    train_y.append(char_to_idx[next_char])

train_X = torch.tensor(train_X, dtype=torch.long)
train_y = torch.tensor(train_y, dtype=torch.long)

##### Model Definition #####
class LSTMPoetryGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMPoetryGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # We only need the last time step
        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden and cell states to zeros
        return (torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device))

##### Hyperparameters #####
input_size = char_size  # Number of unique characters
hidden_size = 300       # Adjusted to capture poetry structure
output_size = char_size
num_epochs = 20        # Number of training epochs
learning_rate = 0.001   # Learning rate for optimizer
batch_size = 32         # Mini-batch size for faster computation

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model, loss function, and optimizer
model = LSTMPoetryGenerator(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

##### Training Function #####
def train(model, train_X, train_y):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i in tqdm(range(0, len(train_X), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_X = train_X[i:i + batch_size].to(device)
            batch_y = train_y[i:i + batch_size].to(device)
            hidden = model.init_hidden(batch_X.size(0))
            
            # Forward pass
            optimizer.zero_grad()
            output, hidden = model(batch_X, hidden)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_X):.4f}")

##### Text Generation Function #####
def generate_text(model, start_text="i", num_chars=100):
    model.eval()
    input_seq = torch.tensor([char_to_idx[c] for c in start_text], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    generated_text = start_text

    with torch.no_grad():
        for _ in range(num_chars):
            output, hidden = model(input_seq, hidden)
            predicted_char_idx = output.argmax(dim=1).item()
            predicted_char = idx_to_char[predicted_char_idx]
            generated_text += predicted_char
            input_seq = torch.tensor([[predicted_char_idx]], dtype=torch.long).to(device)

    return generated_text

##### Training and Testing #####
# Train the model
train(model, train_X, train_y)

# Generate poetry using the trained model
generated_poetry = generate_text(model, start_text="i", num_chars=100)
print("Generated Poetry:\n", generated_poetry)
