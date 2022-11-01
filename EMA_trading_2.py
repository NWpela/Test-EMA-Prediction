import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from indicators import compute_baseline, compute_EMA
import math as m

matplotlib.use('Qt5Agg')

"""
    This file aims test the next EMA trading strategies with quantity of asset controlled to follow the residual
"""

# --- Parameters Definition ---

ASSET = "BTC"
FIELD = "Close"
N_TOT = 10000
N_OFFSET = 4000

EWA_WINDOWS_LIST = [(1, 30)]

initial_mtm = 100
Q0_prop = 0.5
epsilon = 0.0/100
delta = 0.02
epsilon_coeff = 1

N_steps = 10000


# --- Data Loading, Indicators ,Normalisation and baselines---

raw_df = pd.read_csv(f"binance_data/{ASSET}EUR_15m_v1.csv", sep=';')[N_OFFSET:N_OFFSET+N_TOT].reset_index()

for ewa_windows in EWA_WINDOWS_LIST:
    baseline_name = f"BASELINE_{ewa_windows[0]}_{ewa_windows[1]}"
    compute_baseline(raw_df, ewa_windows[0], ewa_windows[1], baseline_name)
# compute residual
compute_EMA(raw_df, EWA_WINDOWS_LIST[-1][1])

# Computation

cash_tot = (1 - Q0_prop) * initial_mtm
Q0 = Q0_prop * initial_mtm / raw_df["Close"][0]  # = Q0
Q_tot = Q0
MtM_tot = initial_mtm
b_name = f"BASELINE_{EWA_WINDOWS_LIST[0][0]}_{EWA_WINDOWS_LIST[0][1]}"
R_name = "RESIDUAL"

# hehe test
#raw_df["RESIDUAL"] = raw_df["RESIDUAL"].mean()
#raw_df["RESIDUAL"] = raw_df["RESIDUAL"].mean() * np.array([1 + 0.5 * m.sin(4.5*m.pi/N_TOT*i) for i in range(N_TOT)])
#raw_df["Close"] = raw_df["RESIDUAL"] + raw_df[b_name]

#R_ref = sum(raw_df["RESIDUAL"][:N_steps])/N_steps
initial_asset_eq = 100/raw_df["Close"][0]
sigma_b = raw_df[b_name].std()

# compute differences
raw_df["DELTA_S"] = raw_df["Close"].diff()
raw_df["DELTA_R"] = raw_df["RESIDUAL"].diff()

MtM_list = [MtM_tot]
cash_list = [cash_tot]
Q_list = [Q_tot]
Q_val_list = [MtM_tot - cash_tot]
index_list = [MtM_tot]
R_index_list = [MtM_tot]
buy_sell_ind_list = [0]

for i in range(1, N_steps):
    data_t = raw_df.iloc[i]
    S = data_t["Close"]
    R = data_t[R_name]
    delta_S = data_t["DELTA_S"]
    delta_R = data_t["DELTA_R"]

    # testing conditions for buy/sell
    delta_Q = (Q0 * delta_R - Q_tot * delta_S) / S  # formula to reproduce the EMA variations
    if delta_Q > 0:  # buy
        # fees adjustment when buying
        delta_Q /= (1-epsilon)

        cash_tot -= S * delta_Q
        Q_tot += delta_Q * (1-epsilon)

        buy_sell_ind_list.append(1)
    elif delta_Q < 0:  # sell
        cash_tot -= delta_Q * S * (1 - epsilon)
        Q_tot += delta_Q

        buy_sell_ind_list.append(-1)
    else:
        buy_sell_ind_list.append(0)

    MtM_tot = cash_tot + S * Q_tot
    Q_val_tot = S * Q_tot

    MtM_list.append(MtM_tot)
    cash_list.append(cash_tot)
    Q_list.append(Q_tot)
    index_list.append(initial_asset_eq * S)
    R_index_list.append(initial_asset_eq * R)
    Q_val_list.append(Q_val_tot)


# Plotting

# main plot
plt.figure()
plt.plot(np.array(MtM_list))
plt.plot(np.array(index_list))
plt.plot(np.array(R_index_list))
plt.plot(2 * (np.array(Q_val_list) - 50) + initial_mtm)

# cash
plt.figure()
plt.plot(np.array(cash_list))
avg_period = 200
average_cash_list = []
for i in range(len(cash_list)):
    if i < avg_period:
        average_cash_list.append(sum(cash_list[:i+1])/(i+1))
    else:
        average_cash_list.append(sum(cash_list[i - avg_period + 1:i + 1]) / avg_period)
plt.plot(np.array(average_cash_list))

# Q
plt.figure()
plt.plot(np.array(Q_list))

# performance
plt.figure()
plt.plot(np.array(MtM_list) - np.array(index_list))
