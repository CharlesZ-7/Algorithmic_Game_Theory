import os
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split

class CampaignPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CampaignPredictor, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_normal_(param.data)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)  # Apply the fully connected layer to every time step
        return torch.sigmoid(out)  # Apply sigmoid activation to every output

class SequenceBuffer:
    def __init__(self, sequence_length, feature_size):
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.buffer = torch.zeros((1, sequence_length, feature_size))

    def update_buffer(self, new_data):
        self.buffer = torch.roll(self.buffer, -1, dims=1)
        self.buffer[0, -1, :] = new_data

    def get_buffer(self):
        return self.buffer.clone()

def train(model, optimizer, train_loader, epochs, device, checkpoint_interval=5):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save checkpoint at specified interval and the last epoch
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'loss': loss.item()
            }, filename="checkpoint.pth")
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')



def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    print(f'Average Loss: {total_loss / len(test_loader):.4f}')

def save_checkpoint(state, filename="checkpoint.pth"):
    """Save model and optimizer states to a file."""
    torch.save(state, filename)
    print(f"Checkpoint saved at '{filename}'")  # Diagnostic message
    
def load_checkpoint(model, filename="checkpoint.pth"):
    """Load model and optimizer states from a file."""
    import os
    if not os.path.exists(filename):
        print(f"No checkpoint found at '{filename}'. Starting from scratch.")
        return 0, float('inf')  # Default values if no checkpoint exists
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state'])
    return checkpoint['epoch'], checkpoint['loss']


def main():
    # Load your data here
    with open("data.json", "r") as f:
        data = json.load(f)

    features = torch.tensor(data['features']).float()
    labels = torch.tensor(data['labels']).float()

    dataset = TensorDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CampaignPredictor(input_size=26, hidden_size=100, num_layers=3, output_size=26)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=0.001)

    # Load a pre-existing checkpoint
    epoch_start, loss = load_checkpoint(model, filename="checkpoint.pth")
    if epoch_start > 0:
        print(f"Resuming from epoch {epoch_start} with loss {loss}")
    else:
        print("No valid checkpoint found; initializing training from the beginning.")

    train(model, optimizer, train_loader, epochs=10, device=device, checkpoint_interval=1)
    evaluate(model, test_loader, device=device)

if __name__ == '__main__':
    main()
