import torch.nn as nn
import torch.nn.functional as F

class FeedForwardExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x