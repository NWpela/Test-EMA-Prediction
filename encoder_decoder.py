import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
from indicators import compute_baseline
from models import LSTM_baseline_predictive, LSTM_baseline_lite
import random
import string
import datetime as dt

"""
    This file aims to use the encoder/decoder architecture to extract the key nuber of features of the binance datasets
"""


# --- Data Loading ---

ASSET = "BTC"
FIELDS = [

]

#### To be continued

# encoder
class LSTM_baseline_lite(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nn_inter_dim, tail_length):
        super(LSTM_baseline_lite, self).__init__()
        # 2 classes: one for the base value and the second one for the exponential multiplier
        # pred = base_value * exp(exp_value)
        self.num_classes = 2
        self.input_size = input_size
        self.num_layers = num_layers  # number of layers
        self.hidden_size = hidden_size  # hidden state
        self.nn_inter_dim = nn_inter_dim
        self.tail_length = tail_length

        self.relu = nn.ReLU()

        # lstm
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        # tail
        self.tail_items = []
        for i_layer in range(tail_length):
            if i_layer > 0:
                in_size = nn_inter_dim
            else:
                in_size = hidden_size
            if i_layer < tail_length - 1:
                out_size = nn_inter_dim
            else:
                out_size = self.num_classes
            self.tail_items.append(self.relu)
            self.tail_items.append(nn.Linear(in_size, out_size))
        self.tail_sequential = nn.Sequential(*self.tail_items)

    def forward(self, x_in):
        # LSTM
        h_0 = torch.zeros(self.num_layers, x_in.size(0), self.hidden_size)  # hidden state
        h_0.requires_grad = False
        c_0 = torch.zeros(self.num_layers, x_in.size(0), self.hidden_size)  # internal state
        c_0.requires_grad = False
        lstm_out, (self.hn, self.cn) = self.lstm(x_in, (c_0, h_0))  # lstm with input, hidden, and internal state

        # tail
        lstm_out = lstm_out.reshape(-1, self.hidden_size)  # reshaping the data for Dense layer next
        tail_out = self.tail_sequential(lstm_out)

        # pred = base_value * exp(exp_value)
        pred = (tail_out[:, [0]] * torch.exp(tail_out[:, [1]]))

        return pred