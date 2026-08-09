import numpy as np
import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=hidden_units, out_channels=hidden_units, kernel_size=1
        )
        self.dropout1 = nn.Dropout(dropout_rate=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=hidden_units, out_channels=hidden_units, kernel_size=1
        )
        self.dropout2 = nn.Dropout(dropout_rate=dropout_rate)

    def forward(self, inputs):
        # inputs shape: (batch size, length, channels)
        # conv1d expected shape: (batch size, channels, length)
        x = inputs.transpose(-1, -2)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = x.transpose(-1, -2)
        outputs = x + inputs
        return outputs
