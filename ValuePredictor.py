import torch
import torch.nn as nn

import Constants


class ValuePredictor(nn.Module):
    def __init__(self, input_size=Constants.LSTM_DEFAULT_INPUT_SIZE):
        super(ValuePredictor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=Constants.LSTM_HIDDEN_SIZE, num_layers=Constants.LSTM_NUM_LAYERS, bidirectional=False, batch_first=True)
        self.linear = nn.Linear(in_features=Constants.LSTM_HIDDEN_SIZE, out_features=Constants.LINEAR_LAYER)

    def forward(self, histories):
        histories = histories.to(self.device)
        histories.requires_grad_(True)
        histories = self.lstm(histories)[0][:, -1]
        histories = self.linear(histories).squeeze(1)
        return histories
