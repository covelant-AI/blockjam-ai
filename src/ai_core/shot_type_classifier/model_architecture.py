import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming you have the same model class definition (GRUModel)
# If not, you need to define it here with the exact same architecture as the one used for training.

class GRUModel(nn.Module):
    def __init__(self, model_path, device, input_size=26, seq_len=30, hidden_size=128, num_layers=3, num_classes=4, dropout=0.1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # dropout only for >1 layer
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        
        self.fc3 = nn.Linear(16, num_classes)

        self.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.eval()

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        _, h = self.gru(x)  # h shape: (num_layers, batch, hidden_size)
        h = h[-1]           # Take last layer's hidden state: (batch, hidden_size)
        h = self.dropout(h)

        h = self.fc1(h)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)

        h = self.fc2(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.dropout(h)

        out = self.fc3(h)  # raw logits
        
        probs = F.softmax(out, dim=1)  # convert logits to probabilities
        return probs