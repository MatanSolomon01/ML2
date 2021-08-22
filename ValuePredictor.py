import torch
import torch.nn as nn

import Constants


class ValuePredictor(nn.Module):
    def __init__(self, input_size=Constants.LSTM_DEFAULT_INPUT_SIZE):
        super(ValuePredictor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=Constants.LSTM_HIDDEN_SIZE, num_layers=Constants.LSTM_NUM_LAYERS, bidirectional=False, batch_first=True)
        self.linear = nn.Linear(in_features=Constants.LSTM_HIDDEN_SIZE, out_features=Constants.LINEAR_LAYER)

    def forward(self, other_things):
        other_things = other_things.to(self.device)
        other_things.requires_grad_(True)
        other_things = self.lstm(other_things)[0][:, -1]
        other_things = self.linear(other_things).squeeze(1)
        return other_things
