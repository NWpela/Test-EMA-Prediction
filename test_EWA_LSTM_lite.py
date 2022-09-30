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

matplotlib.use('Qt5Agg')

"""
    This file aims to use the previous model from EMD decomposition and use it with EWA decomposition instead
"""

# --- Parameters Definition ---

FEATURE_SIZE = 50
SEQ_SIZE = 200
BATCH_SIZE_TRAINING = 16
BATCH_SIZE_TESTING = 7
ASSET = "BTC"
FIELD = "Close"

EWA_WINDOWS = 1, 10

NUM_EPOCHS = 500
LEARNING_RATE = 0.001
LOSS_VAR_FLOOR = -1
WITH_STD = True
STD_COEFF = 0.001
ONE_BATCH = False

HIDDEN_SIZE = 30
NN_INTER_DIM = 10
NUM_LAYERS = 2
TAIL_LENGTH = 3
# D:/nw-trading/test_EWA_prediction
SAVE_PATH = r"saved_models/"

N_training = BATCH_SIZE_TRAINING * SEQ_SIZE + FEATURE_SIZE - 1
N_tot = (BATCH_SIZE_TRAINING + BATCH_SIZE_TESTING) * SEQ_SIZE + FEATURE_SIZE - 1

# --- Data Loading, Indicators ,Normalisation and EMD ---

raw_df = pd.read_csv(f"binance_data/{ASSET}EUR_15m_v1.csv", sep=';')

BASELINE_NAME = f"BASELINE_{EWA_WINDOWS[0]}_{EWA_WINDOWS[1]}"
compute_baseline(raw_df, EWA_WINDOWS[0], EWA_WINDOWS[1], BASELINE_NAME)

series = raw_df[BASELINE_NAME][-N_tot-1:]
bline = np.array(series / series.std())


# --- Imfs Sequencing For Training ---

bline_x = []
bline_y = []
for j in range(FEATURE_SIZE - 1, N_training):
    bline_x.append(bline[j - FEATURE_SIZE + 1:j + 1])
    bline_y.append([bline[j + 1]])
bline_x = np.array(bline_x)
bline_y = np.array(bline_y)
if ONE_BATCH:
    bline_x = bline_x.reshape((1, BATCH_SIZE_TRAINING * SEQ_SIZE, FEATURE_SIZE))
else:
    bline_x = bline_x.reshape((BATCH_SIZE_TRAINING, SEQ_SIZE, FEATURE_SIZE))
bline_y = bline_y.reshape((BATCH_SIZE_TRAINING * SEQ_SIZE, 1))

bline_x = torch.Tensor(bline_x)
bline_y = torch.Tensor(bline_y)


# --- Imfs Sequencing For Testing ---

bline_test_x = []
bline_test_y = []
for j in range(FEATURE_SIZE - 1, N_tot):
    bline_test_x.append(bline[j - FEATURE_SIZE + 1:j + 1])
    bline_test_y.append([bline[j + 1]])
bline_test_x = np.array(bline_test_x)
bline_test_y = np.array(bline_test_y)
if ONE_BATCH:
    bline_test_x = bline_test_x.reshape((1, (BATCH_SIZE_TRAINING + BATCH_SIZE_TESTING) * SEQ_SIZE, FEATURE_SIZE))
    bline_test_y = bline_test_y.reshape((1, (BATCH_SIZE_TRAINING + BATCH_SIZE_TESTING) * SEQ_SIZE, 1))
else:
    bline_test_x = bline_test_x.reshape(((BATCH_SIZE_TRAINING + BATCH_SIZE_TESTING), SEQ_SIZE, FEATURE_SIZE))
    bline_test_y = bline_test_y.reshape(((BATCH_SIZE_TRAINING + BATCH_SIZE_TESTING), SEQ_SIZE, 1))

bline_test_x = torch.Tensor(bline_test_x)
bline_test_y = torch.Tensor(bline_test_y)

dataY_plot = bline_test_y.data.numpy().reshape((-1, 1))


# --- Build ---

lstm_baseline_lite = LSTM_baseline_predictive(FEATURE_SIZE, HIDDEN_SIZE, NUM_LAYERS, NN_INTER_DIM, TAIL_LENGTH)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm_baseline_lite.parameters(), lr=LEARNING_RATE)


# --- Epoch Loop ---

# this must only be used for multi batch mode

def train():
    all_losses = []
    prev_loss_value = 1e-7
    for epoch in range(NUM_EPOCHS):
        output = lstm_baseline_lite.forward(bline_x)  # forward pass
        optimizer.zero_grad()  # calculate the gradient, manually setting to 0

        # obtain the loss function
        loss = criterion(output, bline_y)
        if WITH_STD:
            loss += STD_COEFF * (output - bline_y).std()

        loss.backward()  # calculates the loss of the loss function

        optimizer.step()  # improve from loss, i.e. backprop

        loss_value = loss.item()
        all_losses.append(loss_value)
        if epoch % 10 == 0:
            loss_var_value = abs((loss_value - prev_loss_value) / prev_loss_value)
            prev_loss_value = loss_value
            print("Epoch: %d, loss: %1.8f, loss var: %1.5f" % (epoch, loss_value, loss_var_value))
            if loss_var_value < LOSS_VAR_FLOOR:
                break

    # loss plot
    plt.figure(1)
    plt.plot(np.array(all_losses))
    print(f"Last loss value: {all_losses[-1]}")

# --- Plotting ---

def plot_model():
    train_predict = lstm_baseline_lite(bline_test_x.reshape((1, -1, FEATURE_SIZE)))
    data_predict = train_predict.data.numpy().reshape((-1, 1))
    plt.figure(2)  # size of the training set
    plt.axvline(x=N_training, c='r', linestyle='--')
    plt.plot(dataY_plot, label=f'Actual data')
    plt.plot(data_predict, label=f'Predicted data')
    plt.legend()
    plt.show()
    MAPE = abs((data_predict - dataY_plot) / dataY_plot).mean()
    print(f"MAPE: {MAPE}")


# --- Saving / Loading ---

def save_model():
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    new_id = ASSET + '_' + BASELINE_NAME + '_' + ''.join(random.choice(letters) for _ in range(10))
    line_model = dict(
        ID=new_id,
        BASELINE=BASELINE_NAME,
        DATE=dt.datetime.now().strftime("%d/%m/%Y"),
        FEATURE_SIZE=FEATURE_SIZE,
        SEQ_SIZE=SEQ_SIZE,
        BATCH_SIZE_TRAINING=BATCH_SIZE_TRAINING,
        BATCH_SIZE_TESTING=BATCH_SIZE_TESTING,
        ASSET=ASSET,
        FIELD=FIELD,
        NUM_EPOCHS=NUM_EPOCHS,
        LEARNING_RATE=LEARNING_RATE,
        LOSS_VAR_FLOOR=LOSS_VAR_FLOOR,
        WITH_STD=WITH_STD,
        STD_COEFF=STD_COEFF,
        ONE_BATCH=ONE_BATCH,
        HIDDEN_SIZE=HIDDEN_SIZE,
        NN_INTER_DIM=NN_INTER_DIM,
        NUM_LAYERS=NUM_LAYERS,
        TAIL_LENGTH=TAIL_LENGTH
    )

    if os.path.isfile("lstm_models.csv"):
        df_models = pd.read_csv("lstm_models.csv", sep=';')
        df_models = pd.concat([df_models, pd.Series(line_model).to_frame().transpose()])
    else:
        df_models = pd.Series(line_model).to_frame().transpose()

    df_models.to_csv("lstm_models.csv", sep=';', index=False)
    torch.save(lstm_baseline_lite.state_dict(), SAVE_PATH + new_id + '.pt')


def load_model(file_name_code):
    lstm_baseline_lite.load_state_dict(torch.load(SAVE_PATH + file_name_code))


# --- Execution ---

if __name__ == "__main__":
    train()
    plot_model()
