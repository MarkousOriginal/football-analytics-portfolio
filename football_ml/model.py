"""LSTM encoder with an additive attention query projected from FIFA attributes."""
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out, fifa_vec):
        scores = self.v(torch.tanh(self.W(lstm_out) + fifa_vec.unsqueeze(1)))
        weights = torch.softmax(scores, dim=1)
        return (weights * lstm_out).sum(dim=1)


class RNNWithFIFAAttention(nn.Module):
    def __init__(self, seq_input_dim, static_input_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(seq_input_dim, hidden_dim, batch_first=True)
        self.fifa_proj = nn.Sequential(nn.Linear(static_input_dim, 64), nn.ReLU(), nn.Linear(64, hidden_dim))
        self.attn = Attention(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, seq_x, fifa_x):
        lstm_out, _ = self.lstm(seq_x)
        return self.head(self.attn(lstm_out, self.fifa_proj(fifa_x)))
