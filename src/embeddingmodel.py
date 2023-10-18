import torch
from torch import nn


# class LSTMModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.linear = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         _, (h, _) = self.lstm(x)
#         out = self.linear(h[-1])
#         return out
