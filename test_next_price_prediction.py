import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
from models import LSTM_baseline_predictive
from indicators import compute_baseline, compute_EMA


matplotlib.use('Qt5Agg')

"""
    This file aims to build a full model to predict the next prices of a stock using EMA decomposition
    This looks like what has been done with EMD but it is simplified
    It is also possible to predict trend for at a given point for a given horizon
"""


# --- Global parameters ---

ASSET = "BTC"
FEATURE_SIZE = 50
SEQ_SIZE = 200
BATCH_SIZE_USED = 3  # batch size used for prediction (for lstm)
BATCH_SIZE_TOT = 16  # total batch size available
N_USED_RESIDUALS = 10  # number of used residual values to predict the next ones

EMA_WINDOWS_LIST = [(1, 10), (10, 70), (70, 200)]

# each lstm num is mapped to a model id in the lstm_models.csv file
BASELINE_NAMES = [f"BASELINE_{ewa_windows[0]}_{ewa_windows[1]}" for ewa_windows in EMA_WINDOWS_LIST]
BASELINE_MODELS_IDS = [
    "BTC_BASELINE_1_10_bbuxstidjh",
    "BTC_BASELINE_10_70_nxyeilrdur",
    "BTC_BASELINE_70_200_ydeteqszyf"
]
N_baselines = len(EMA_WINDOWS_LIST)
BASELINE_STDS = [0 for _ in range(N_baselines)]

SAVE_PATH = r"saved_models/"

# N_tot is the number of data from the spots used in the simulation (loaded from the csv file)
N_tot = BATCH_SIZE_TOT * SEQ_SIZE + FEATURE_SIZE - 1
# N_used is the number of sequences used by all the models at each iteration
N_used = BATCH_SIZE_USED * SEQ_SIZE + FEATURE_SIZE - 1

# simulation indexes is initialized
# start / end indexes are by default the last index of start / end sequences in the whole df
i_simul_start = FEATURE_SIZE - 1
i_simul_end = N_used - 1

MODELS_LIST = []

# --- Definition the model for each imf ---

# the intermediate ones by the lstm model
for i in range(N_baselines):
    # looking for the model via the id
    df_models = pd.read_csv("lstm_models.csv", sep=';')
    model_id = BASELINE_MODELS_IDS[i]
    model_param = df_models[df_models.ID == model_id].iloc[0]

    input_size = model_param.FEATURE_SIZE
    hidden_size = model_param.HIDDEN_SIZE
    num_layers = model_param.NUM_LAYERS
    nn_inter_dim = model_param.NN_INTER_DIM
    tail_length = model_param.TAIL_LENGTH

    # creating the model
    model = LSTM_baseline_predictive(input_size, hidden_size, num_layers, nn_inter_dim, tail_length)
    # loading the parameters
    model.load_state_dict(torch.load(SAVE_PATH + model_id + ".pt"))
    print(f"Model {model_id + '.pt'} loaded")
    MODELS_LIST.append(model)


# --- Data Loading and Normalisation ---

raw_df = pd.read_csv(f"binance_data/{ASSET}EUR_15m_v1.csv", sep=';')[:N_tot+1]
# compute required baselines
for ewa_windows in EMA_WINDOWS_LIST:
    baseline_name = f"BASELINE_{ewa_windows[0]}_{ewa_windows[1]}"
    compute_baseline(raw_df, ewa_windows[0], ewa_windows[1], baseline_name)
# compute residual
compute_EMA(raw_df, EMA_WINDOWS_LIST[-1][1])

# normalisation
for i in range(N_baselines):
    bline = BASELINE_NAMES[i]
    std = raw_df[bline].std()
    raw_df[bline] = raw_df[bline] / std
    BASELINE_STDS[i] = std


# --- Computation functions ---

def get_next_residual_predictions(n_ahead: int = 1):
    x_base = np.arange(-N_USED_RESIDUALS+1, 1)
    y_base = np.array(raw_df["Close"][i_simul_end - N_USED_RESIDUALS + 1: i_simul_end+1])
    p = np.polyfit(x_base, y_base, deg=3)
    next_vals = np.polyval(p, np.arange(1, n_ahead+1))
    return next_vals


def get_next_y_values(n_ahead: int) -> np.array:
    return np.array(raw_df["Close"][i_simul_end+1:i_simul_end+n_ahead+1])


def get_sequenced_data_list_lstm() -> list:
    # Computes the list of sequenced data for each baseline
    bline_list_x = []
    for k in range(N_baselines):
        bline_x = []
        for j in range(i_simul_start, i_simul_end + 1):
            # the j+1 value is predicted, so it goes up to j with length=FEATURE_SIZE
            bline_x.append(raw_df[BASELINE_NAMES[i]][j - FEATURE_SIZE + 1:j + 1])
        bline_list_x.append(torch.unsqueeze(torch.Tensor(bline_x), 0))
    return bline_list_x


def get_predictions(n_ahead: int):
    next_tot_prediction = np.zeros(n_ahead)
    # baselines
    baseline_sequences = get_sequenced_data_list_lstm()
    for k in range(N_baselines):
        pred_bline = MODELS_LIST[k].predict(baseline_sequences[k], n_ahead) * BASELINE_STDS[k]
        next_tot_prediction += pred_bline
    # residual
    next_tot_prediction += get_next_residual_predictions(n_ahead=n_ahead)

    # get the base sequence prediction (forward result)
    # baselines
    base_prediction = np.zeros(BATCH_SIZE_USED * SEQ_SIZE)
    for k in range(N_baselines):
        base_prediction += (MODELS_LIST[k](baseline_sequences[k]).reshape(-1) * BASELINE_STDS[k]).detach().numpy()
    # residual (1st order approximation)
    prices = np.array(raw_df["Close"][i_simul_start:i_simul_end+1])
    shifted_prices = np.array(raw_df["Close"][i_simul_start:i_simul_end+1].shift(periods=1, fill_value=prices[0]))  # filled by the first value
    base_prediction += 2 * prices - shifted_prices

    return base_prediction, next_tot_prediction


def plot_prediction(n_ahead: int):
    x_pred = np.arange(i_simul_end+1, i_simul_end+n_ahead+1)
    x_extended = np.arange(i_simul_start, i_simul_end+n_ahead+1)
    plt.figure()

    # plot spot (from beginning to predicted values)
    y_spot = np.array(raw_df["Close"][i_simul_start:i_simul_end+n_ahead+1])
    plt.plot(x_extended, y_spot)

    # plot prediction
    y_model, y_pred = get_predictions(n_ahead)
    plt.plot(x_pred, y_pred)

    # plot model fitting
    x_model = np.arange(i_simul_start + 1, i_simul_end + 2)  # 1-value offset because next value is approximated
    plt.plot(x_model, y_model)

    # plot line to separate pred and previous data
    plt.axvline(x=i_simul_end+0.5, c='r', linestyle='--')

    plt.plot()
