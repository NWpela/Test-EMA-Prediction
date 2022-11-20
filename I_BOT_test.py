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
import math as m

matplotlib.use('Qt5Agg')

"""
    This file aims to create the base model for Cobalt with RL model and key features
"""

# --- Parameters Definition ---

ASSET = "BTC"
FEES = 0.1/100

FEATURE_SIZE = 50
SEQ_SIZE = 200
N_SEQ_TRAIN = 30
N_SEQ_TEST = 10
SEQUENCING_INCREMENT = 200

EWA_WINDOWS = 1, 10

EPOCHS_PER_SEQ = 10
N_SEQ_ITER_TRAIN = 200  # length of the training in terms of sequences
LEARNING_RATE = 0.001
GAMMA = 0.99
ENTROPY_COEFF = 0.0001
MU_COEFF = 0.

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
            nn.Linear(self.lstm_hidden_size+2, 5),
            nn.ReLU(),
            nn.Linear(5, 1, bias=False),  # set bias = False because returns always bias in other case
            nn.Tanh()
        )

        # tail var
        self.tail_var_act = nn.Sequential(
            nn.Linear(self.lstm_hidden_size+2, 5),
            nn.ReLU(),
            nn.Linear(5, 1, bias=False),  # same principle with var
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
            nn.Linear(self.lstm_hidden_size+2, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, market_features_in, cash_prop_t, asset_t, h_act_t_1, c_act_t_1, h_crit_t_1, c_crit_t_1):
        # Actor
        # conv1d
        conv_act_out = self.conv_act(market_features_in)
        # lstm
        h_act_t, c_act_t = self.lstm_act(conv_act_out[0], (h_act_t_1, c_act_t_1))
        lstm_act_out = h_act_t.clone()
        #h_act_t, c_act_t = h_act_t.clone().detach(), c_act_t.clone().detach()
        # concat with cash and asset
        tail_act_in = torch.cat([lstm_act_out, torch.Tensor(np.array([[cash_prop_t, asset_t]]))], dim=1)
        # tail
        mu_act, var_act = self.tail_mu_act(tail_act_in), self.tail_var_act(tail_act_in)
        
        # Critic
        # conv1d
        conv_crit_out = self.conv_crit(market_features_in)
        # lstm
        h_crit_t, c_crit_t = self.lstm_crit(conv_crit_out[0], (h_crit_t_1, c_crit_t_1))
        lstm_crit_out = h_crit_t.clone()
        #h_crit_t, c_crit_t = h_crit_t.clone().detach(), c_crit_t.clone().detach()
        # concat with cash and asset
        tail_crit_in = torch.cat([lstm_crit_out, torch.Tensor(np.array([[cash_prop_t, asset_t]]))], dim=1)
        # tail
        val_crit = self.tail_val_crit(tail_crit_in)

        return mu_act[0], var_act[0], val_crit[0], (h_act_t, c_act_t), (h_crit_t, c_crit_t)


def calc_log_prob(mu_v_n: torch.Tensor, var_v_n: torch.Tensor, actions_v: np.array) -> torch.Tensor:
    actions_t = torch.FloatTensor(actions_v)
    # Returns the log of the normal distribution as a tensor for vector-like inputs
    return - ((mu_v_n - actions_t) ** 2) / (2*var_v_n.clamp(min=1e-3)) - torch.log(torch.sqrt(2 * m.pi * var_v_n.clamp(min=1e-3)))


# --- Build ---

i_bot_model = I_BOT_model(FEATURE_SIZE, 4)

criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(i_bot_model.parameters(), lr=LEARNING_RATE)


# --- Training ---

MtM_df = pd.DataFrame(columns=["SEQUENCE", "EPOCH", "MtM_diff_index", "Loss"])
all_mu_list = []
all_MtM_list = []
all_h_c_list = []
prev_loss_value = 1

for n in range(N_SEQ_ITER_TRAIN):
    seq_num = random.choice(range(N_SEQ_TRAIN))
    print(f"SEQUENCE ITERATION {n}: n°{seq_num} chosen")
    seq_features = training_feature_seq_list[seq_num]
    seq_prices = training_prices_list[seq_num]

    for i in range(EPOCHS_PER_SEQ):
        h_c_list = []
        mu_list = []
        MtM_list = [1]
        mu_n_list = []
        log_probs_n = []
        values_n = []
        values = []
        rewards = []
        entropy_loss = torch.FloatTensor([0])

        MtM = 1
        cash = random.random()
        asset = (1 - cash) / seq_prices[0]
        index_asset = 1 / seq_prices[0]
        MtM_diff_index = 0

        h_act, c_act = torch.zeros(1, i_bot_model.lstm_hidden_size), torch.zeros(1, i_bot_model.lstm_hidden_size)
        h_crit, c_crit = torch.zeros(1, i_bot_model.lstm_hidden_size), torch.zeros(1, i_bot_model.lstm_hidden_size)

        # (requires grad ?)

        for j in range(SEQ_SIZE-1):  # SEQ_SIZE-1 because j+1 value is used for reward computation
            # initial MtM computations
            MtM = cash + asset * seq_prices[j]
            MtM_index = index_asset * seq_prices[j]
            MtM_diff_index = MtM - MtM_index

            market_features_in = torch.FloatTensor(np.array([seq_features[j]]))
            mu_act_n, var_act_n, val_crit_n, (h_act_t, c_act_t), (h_crit_t, c_crit_t) =\
                i_bot_model.forward(market_features_in, cash, asset, h_act, c_act, h_crit, c_crit)
            h_act, c_act, h_crit, c_crit = h_act_t, c_act_t, h_crit_t, c_crit_t
            h_c_list.append((h_act.detach(), c_act.detach()))

            value = val_crit_n.detach().numpy()[0]
            mu = mu_act_n.detach().numpy()
            sigma = torch.sqrt(var_act_n).detach().numpy()

            delta_raw = np.random.normal(mu, sigma)
            log_prob_n = calc_log_prob(mu_act_n, var_act_n, delta_raw)
            entropy_loss += (-(torch.log(2 * m.pi * var_act_n.clamp(min=1e-3)) + 1) / 2)
            delta = delta_raw[0].clip(-1, 1)  # can be changed

            # taking action
            if delta > 0:
                # buy
                asset += delta * cash / seq_prices[j] * (1 - FEES)
                cash -= delta * cash
            else:
                # sell
                cash += -delta * asset * seq_prices[j] * (1 - FEES)
                asset -= -delta * asset

            # computes reward using next step
            next_MtM = cash + seq_prices[j+1] * asset
            next_MtM_index = seq_prices[j+1] * index_asset
            next_MtM_diff_index = next_MtM - next_MtM_index
            reward = next_MtM_diff_index - MtM_diff_index  # pnl - pnl_index

            rewards.append(reward)
            values.append(value)
            values_n.append(val_crit_n)
            log_probs_n.append(log_prob_n)
            mu_n_list.append(mu_act_n)

            mu_list.append(mu[0])
            MtM_list.append(MtM)
            all_h_c_list.append(h_c_list)
            #if j % 20 == 0:
            #    print(f"Step {j} - Mu: {mu}, Sigma: {sigma}, Delta: {delta}")

        last_features = torch.FloatTensor(np.array([seq_features[SEQ_SIZE-1]]))
        _, _, Qval, _, _ = i_bot_model.forward(last_features, cash, asset, h_act, c_act, h_crit, c_crit)
        Qval = Qval.detach().numpy()[0]

        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        # compute tensors
        values = torch.FloatTensor(np.array(values))
        Qvals = torch.FloatTensor(Qvals)
        log_probs_n = torch.stack(log_probs_n)

        # advantage and loss
        advantage = Qvals - values
        actor_loss = (-log_probs_n * advantage).mean()
        values_n = torch.FloatTensor(values_n)
        critic_loss = 0.5 * (Qvals - values_n).pow(2).mean()
        mu_loss = torch.std(torch.stack(mu_n_list))
        ac_loss = actor_loss + critic_loss - ENTROPY_COEFF * entropy_loss - MU_COEFF * mu_loss

        # grad
        optimizer.zero_grad()
        ac_loss.backward()
        optimizer.step()

        # scores and log
        loss_value = ac_loss.item()
        loss_distribution = round(actor_loss.item(), 5), round(critic_loss.item(), 5), round(-ENTROPY_COEFF*entropy_loss.item(), 5)
        loss_var_value = (loss_value - prev_loss_value) / prev_loss_value
        prev_loss_value = loss_value
        print("Sequence: %d, Epoch: %d, loss: %1.8f, loss var: %1.5f, MtM_diff_index: %1.5f"
              % (seq_num, i, loss_value, loss_var_value, MtM_diff_index),
              f"Loss distribution: {loss_distribution}")

        dict_score = {"SEQUENCE": seq_num, "EPOCH": i, "MtM_diff_index": MtM_diff_index, "Loss": loss_value}
        MtM_df = pd.concat([MtM_df, pd.Series(dict_score).to_frame().transpose()])

        all_mu_list.append(mu_list)
        all_MtM_list.append((seq_num, MtM_list))

all_mu_list = np.array(all_mu_list)

print("DONE")
MtM_df = MtM_df.reset_index()
MtM_df["MtM_diff_index_smooth"] = MtM_df["MtM_diff_index"].ewm(alpha=0.1).mean()
