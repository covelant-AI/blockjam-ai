import torch
import torch.nn as nn

# Assuming you have the same model class definition (GRUModel)
# If not, you need to define it here with the exact same architecture as the one used for training.

class GRUModel(nn.Module):
    def __init__(self, model_path, device, input_size=26, seq_len=30, hidden_size=24, num_classes=4, dropout=0.05):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.1 # Note: The warning about dropout requires num_layers > 1.
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.load_state_dict(torch.load(model_path, map_location=device))
        self.eval()

    def forward(self, x):
        _, h = self.gru(x)
        h = h.squeeze(0)
        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        out = self.fc2(h)
        return self.softmax(out)