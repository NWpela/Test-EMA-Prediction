import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from indicators import compute_baseline, compute_EMA, compute_RSI
from models import LSTM_baseline_predictive, LSTM_baseline_lite
import random
import string
import datetime as dt
import utils

matplotlib.use('Qt5Agg')

"""
    This file aims to create the base model for Cobalt with RL model and key features
"""


# --- Parameters Definition ---

ASSET = "BTC"

FEATURE_SIZE = 50
SEQ_SIZE = 200
N_SEQ_TRAIN = 30
N_SEQ_TEST = 10
SEQUENCING_INCREMENT = 100

EWA_WINDOWS = 1, 10

EPOCHS_PER_SEQ = 10
LEARNING_RATE = 0.001

HIDDEN_SIZE = 30
NN_INTER_DIM = 10
NUM_LAYERS = 2
TAIL_LENGTH = 3
# D:/nw-trading/test_EWA_prediction
SAVE_PATH = r"saved_models/"

N_training = N_SEQ_TRAIN * SEQ_SIZE + FEATURE_SIZE - 1
N_testing = N_SEQ_TEST * SEQ_SIZE

# --- Data Loading ---

raw_df = pd.read_csv(f"binance_data/{ASSET}EUR_15m_v1.csv", sep=';')

# baseline computation and normalization by residual (no std here)
baseline_name = f"BASELINE_{EWA_WINDOWS[0]}_{EWA_WINDOWS[1]}"
compute_baseline(raw_df, EWA_WINDOWS[0], EWA_WINDOWS[1], baseline_name)
compute_EMA(raw_df, EWA_WINDOWS[1])
raw_df[baseline_name] = raw_df[baseline_name] / raw_df["RESIDUAL"]

# rsi
compute_RSI(raw_df)
raw_df["RSI"] /= 100

# global normalizations
std_bline = raw_df[baseline_name].std()
raw_df[baseline_name] /= std_bline
mean_nb_of_trades = raw_df["Number_of_trades"].mean()
raw_df["Number_of_trades"] /= mean_nb_of_trades


# --- Data Sequencing ---

data_columns = [baseline_name, "RESIDUAL", "RSI", "Number_of_trades"]
training_data_list = []
testing_data_list = []

training_feature_seq_list = []
testing_feature_seq_list = []
training_prices_list = []
testing_prices_list = []

# training
for n in range(N_SEQ_TRAIN):
    df = raw_df.iloc[n*SEQUENCING_INCREMENT:
                     n*SEQUENCING_INCREMENT + SEQ_SIZE + FEATURE_SIZE].copy()
    # residual normalization by initial value
    df["RESIDUAL"] /= df["RESIDUAL"].iloc[0]
    training_data_list.append(df)
    # add features and prices
    training_feature_seq = []
    training_prices = []
    for i in range(SEQ_SIZE):
        training_feature_seq.append(np.array(df[data_columns].iloc[i:i+FEATURE_SIZE]).transpose())
        training_prices.append(df["Close"].iloc[i+FEATURE_SIZE])
    training_feature_seq_list.append(np.array(training_feature_seq))
    training_prices_list.append(np.array(training_prices))
    
# testing
for n in range(N_SEQ_TEST):
    df = raw_df.iloc[N_SEQ_TEST + n*SEQUENCING_INCREMENT:
                     N_SEQ_TEST + n*SEQUENCING_INCREMENT + SEQ_SIZE + FEATURE_SIZE].copy()
    # residual normalization by initial value
    df["RESIDUAL"] /= df["RESIDUAL"].iloc[0]
    testing_data_list.append(df)
    # add features and prices
    testing_feature_seq = []
    testing_prices = []
    for i in range(SEQ_SIZE):
        testing_feature_seq.append(np.array(df[data_columns].iloc[i:i+FEATURE_SIZE]).transpose())
        testing_prices.append(df["Close"].iloc[i+FEATURE_SIZE])
    testing_feature_seq_list.append(np.array(testing_feature_seq))
    testing_prices_list.append(np.array(testing_prices))


# --- Model ---

class I_BOT_model(nn.Module):
    def __init__(self, input_size, channels_in):
        super(I_BOT_model, self).__init__()
        # conv1d params
        self.kernel_size = 10
        self.lstm_hidden_size = 10
        self.num_classes = 2
        self.input_size = input_size

        # Actor part
        # conv1d
        self.conv_act = nn.Sequential(
            nn.Conv1d(channels_in, 1, self.kernel_size)
        )
        self.conv1d_out_size_act = input_size - self.kernel_size + 1

        # LSTM
        self.lstm_act = nn.LSTMCell(self.conv1d_out_size_act, self.lstm_hidden_size)

        # tail mu
        self.tail_mu_act = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 5),
            nn.ReLU(),
            nn.Linear(self.lstm_hidden_size, 1),
            nn.Tanh()
        )

        # tail var
        self.tail_var_act = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 5),
            nn.ReLU(),
            nn.Linear(self.lstm_hidden_size, 1),
            nn.Softplus()
        )

        # Critic part
        # conv1d
        self.conv_crit = nn.Sequential(
            nn.Conv1d(channels_in, 1, self.kernel_size)
        )
        self.conv1d_out_size_crit = input_size - self.kernel_size + 1

        # LSTM
        self.lstm_crit = nn.LSTMCell(self.conv1d_out_size_crit, self.lstm_hidden_size)

        # tail value
        self.tail_val_crit = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 5),
            nn.ReLU(),
            nn.Linear(self.lstm_hidden_size, 1)
        )

    def forward(self, market_features_in, cash_prop_t, asset_t, h_act_t_1, c_act_t_1, h_crit_t_1, c_crit_t_1):
        # Actor
        # conv1d
        conv_act_out = self.conv_act(market_features_in)
        # lstm
        h_act_t, c_act_t = self.lstm(conv_act_out, (h_act_t_1, c_act_t_1))
        lstm_act_out = h_act_t.clone()
        h_act_t, c_act_t = h_act_t.clone().detach(), c_act_t.clone().detach()
        # concat with cash and asset
        tail_act_in = torch.cat([lstm_act_out, torch.Tensor([[cash_prop_t, asset_t]])])
        # tail
        mu_act, var_act = self.tail_mu_act(tail_act_in), self.tail_var_act(tail_act_in)
        
        # Critic
        # conv1d
        conv_crit_out = self.conv_crit(market_features_in)
        # lstm
        h_crit_t, c_crit_t = self.lstm(conv_crit_out, (h_crit_t_1, c_crit_t_1))
        lstm_crit_out = h_crit_t.clone()
        h_crit_t, c_crit_t = h_crit_t.clone().detach(), c_crit_t.clone().detach()
        # concat with cash and asset
        tail_crit_in = torch.cat([lstm_crit_out, torch.Tensor([[cash_prop_t, asset_t]])])
        # tail
        val_crit = self.tail_val_crit(tail_crit_in)

        return mu_act, var_act, val_crit, (h_act_t, c_act_t), (h_crit_t, c_crit_t)


# --- Training ---

