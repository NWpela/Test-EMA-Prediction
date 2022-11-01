import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from indicators import compute_baseline, compute_EMA
import math as m

matplotlib.use('Qt5Agg')

"""
    This file aims test the EMA trading strategy with multiple baselines
"""

# --- Parameters Definition ---

ASSET = "BTC"
FIELD = "Close"
N_TOT = 30000
N_OFFSET = 0

EWA_WINDOWS_LIST = [(1, 30), (10, 70), (70, 200)]
N_baselines = len(EWA_WINDOWS_LIST)

epsilon = 0.1/100
delta = 0.02
comp_coeff = 1

#N_steps = 1000


# --- Data Loading, Indicators, Normalisation and baselines---

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
b_names_list = [f"BASELINE_{ewa_window[0]}_{ewa_window[1]}" for ewa_window in EWA_WINDOWS_LIST]
R_name = "RESIDUAL"

# hehe test
#raw_df["RESIDUAL"] = raw_df["RESIDUAL"].mean()
#raw_df["RESIDUAL"] = raw_df["RESIDUAL"].mean() * np.array([1 + 0.25 * m.sin(15.5*m.pi/N_TOT*i) for i in range(N_TOT)])
#raw_df["Close"] = raw_df["RESIDUAL"] + raw_df[b_name]

initial_asset_eq = 100/raw_df["Close"][0]
sigma_b_list = [raw_df[b_name].std() for b_name in b_names_list]
sigma_b_sum = sum(sigma_b_list)

# cash and asset allocation
#cash_allocation_list = [sigma_b / sigma_b_sum for sigma_b in sigma_b_list]
#asset_allocation_list = [1 / N_baselines for i in range(N_baselines)]
cash_allocation_list = [0.1, 0.8, 0.1]
asset_allocation_list = [0.1, 0.8, 0.1]

MtM_list = []
cash_list = []
asset_list = []
index_list = []

for i in range(N_TOT):
    data_t = raw_df.iloc[i]
    S = data_t["Close"]
    tot_buy_cash_prop = 0
    tot_sell_asset_prop = 0

    for k in range(N_baselines):
        b = data_t[b_names_list[k]]

        if b > comp_coeff * sigma_b_list[k]:
            # add to sell
            tot_sell_asset_prop += delta * asset_allocation_list[k]
        elif b < - comp_coeff * sigma_b_list[k]:
            # add to buy
            tot_buy_cash_prop += delta * cash_allocation_list[k]

        euro_balance = tot_sell_asset_prop * asset_tot * S - tot_buy_cash_prop * cash_tot
        asset_balance = euro_balance / S
        if euro_balance > 0:
            # sell euro_balance
            asset_tot -= asset_balance
            cash_tot += euro_balance * (1 - epsilon)
        else:
            # buy euro_balance
            asset_tot -= asset_balance * (1 - epsilon)
            cash_tot += euro_balance

    MtM_tot = cash_tot + S * asset_tot

    MtM_list.append(MtM_tot)
    cash_list.append(cash_tot)
    asset_list.append(asset_tot)
    index_list.append(initial_asset_eq * S)


# Plotting

plt.figure()
plt.plot(np.array(MtM_list))
plt.plot(np.array(index_list))

plt.figure()
plt.plot(np.array(MtM_list) - np.array(index_list))

plt.figure()
plt.plot(np.array(cash_list))
