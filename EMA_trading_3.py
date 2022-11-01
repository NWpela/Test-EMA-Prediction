import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from indicators import compute_baseline, compute_EMA
import math as m

matplotlib.use('Qt5Agg')

"""
    This file aims test the EMA trading strategy with limit up/down feature
"""

# --- Parameters Definition ---

ASSET = "BTC"
FIELD = "Close"
N_TOT = 10000
N_OFFSET = 4000

EWA_WINDOWS_LIST = [(1, 200)]

epsilon = 0.1/100
delta = 0.2
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

cash_tot = 100
asset_tot = 0
MtM_tot = 100
b_name = f"BASELINE_{EWA_WINDOWS_LIST[0][0]}_{EWA_WINDOWS_LIST[0][1]}"
R_name = "RESIDUAL"

# hehe test
#raw_df["RESIDUAL"] = raw_df["RESIDUAL"].mean()
#raw_df["RESIDUAL"] = raw_df["RESIDUAL"].mean() * np.array([1 + 0.25 * m.sin(15.5*m.pi/N_TOT*i) for i in range(N_TOT)])
#raw_df["Close"] = raw_df["RESIDUAL"] + raw_df[b_name]

initial_asset_eq = 100/raw_df["Close"][0]
sigma_b = raw_df[b_name].std()


MtM_list = []
cash_list = []
asset_list = []
index_list = []
R_index_list = []
buy_sell_ind_list = []
prop_list = []
nb_trades = 0
buy_flag = True
sell_flag = True

for i in range(N_steps):
    data_t = raw_df.iloc[i]
    S = data_t["Close"]
    b = data_t[b_name]
    R = data_t[R_name]

    # testing conditions for buy/sell
    test_value = b / R
    #test_value = b
    prop = delta

    if test_value > epsilon_coeff * epsilon:
    #if test_value > epsilon_coeff * sigma_b:
        if sell_flag:
            buy_sell_ind_list.append(1)
            nb_trades += 1
            # sell
            cash_tot += S * prop * asset_tot * (1-epsilon)
            asset_tot -= prop * asset_tot
            sell_flag = False
        else:
            buy_sell_ind_list.append(0)
    elif test_value < - epsilon_coeff * epsilon:
    #if test_value < - epsilon_coeff * sigma_b:
        if buy_flag:
            buy_sell_ind_list.append(-1)
            nb_trades += 1
            # buy
            asset_tot += prop * cash_tot * (1-epsilon) / S
            cash_tot -= prop * cash_tot
            buy_flag = False
        else:
            buy_sell_ind_list.append(0)
    else:
        sell_flag = True
        buy_flag = True
        buy_sell_ind_list.append(0)

    MtM_tot = cash_tot + S * asset_tot

    MtM_list.append(MtM_tot)
    cash_list.append(cash_tot)
    asset_list.append(asset_tot)
    index_list.append(initial_asset_eq * S)
    R_index_list.append(initial_asset_eq * R)
    prop_list.append(prop)


# Plotting

plt.figure()
plt.plot(np.array(MtM_list))
plt.plot(np.array(index_list))
plt.plot(np.array(R_index_list))
plt.plot(np.array(buy_sell_ind_list))
#plt.plot(np.array(cash_list))
print(nb_trades/N_steps)

plt.figure()
plt.plot(np.array(MtM_list) - np.array(index_list))

plt.figure()
plt.plot(np.array(prop_list))
