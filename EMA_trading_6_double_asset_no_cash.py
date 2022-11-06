import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from indicators import compute_baseline, compute_EMA
import math as m

matplotlib.use('Qt5Agg')

"""
    This file aims test the EMA trading strategy
"""

# --- Parameters Definition ---

ASSET1 = "BTC"
ASSET2 = "ETH"
N_TOT = 20000
N_OFFSET = 0

EWA_WINDOW = (1, 10)

fees = 0.1/100
delta = 0.01
sigma_coeff = 2


# --- Data Loading, Indicators, Normalisation and baselines---

raw_df1 = pd.read_csv(f"binance_data/{ASSET1}EUR_15m_v1.csv", sep=';')[N_OFFSET:N_OFFSET+N_TOT].reset_index()
raw_df2 = pd.read_csv(f"binance_data/{ASSET2}EUR_15m_v1.csv", sep=';')[N_OFFSET:N_OFFSET+N_TOT].reset_index()

# compute main baselines
baseline_name = f"BASELINE_{EWA_WINDOW[0]}_{EWA_WINDOW[1]}"

# compute cross baseline / residual
df_cross12 = pd.DataFrame()
df_cross12["Close"] = raw_df2.Close / raw_df1.Close  # u12
compute_baseline(df_cross12, EWA_WINDOW[0], EWA_WINDOW[1], baseline_name)
compute_EMA(df_cross12, EWA_WINDOW[1])

# baseline normalization
df_cross12[baseline_name] = df_cross12[baseline_name] / df_cross12["RESIDUAL"]


# --- Computation ---

asset_1 = 50/raw_df1["Close"][0]
asset_2 = 50/raw_df2["Close"][0]
MtM = 100

# initial asset amounts for indexes
initial_asset1_eq = 100/raw_df1["Close"][0]
initial_asset2_eq = 100/raw_df2["Close"][0]
initial_asset_cross_eq1 = 50/raw_df1["Close"][0]
initial_asset_cross_eq2 = 50/raw_df2["Close"][0]

# std
sigma_b_cross_12 = df_cross12[baseline_name].std()

# lists
MtM_list = []
asset1_list = []
asset2_list = []

index1_list = []
index2_list = []
index12_list = []

for i in range(N_TOT):
    asset_1_prev = asset_1
    asset_2_prev = asset_2

    data_t_1 = raw_df1.iloc[i]
    data_t_2 = raw_df2.iloc[i]
    data_t_12 = df_cross12.iloc[i]

    S1 = data_t_1["Close"]
    S2 = data_t_2["Close"]
    S12 = data_t_12["Close"]
    b12 = data_t_12[baseline_name]

    # testing conditions to buy/sell
    # asset 1 vs asset 2
    if b12 > sigma_coeff * sigma_b_cross_12:
        # sell 2 for 1
        asset_1 += S12 * delta * asset_2_prev * (1-fees)
        asset_2 -= delta * asset_2_prev
    elif b12 < - sigma_coeff * sigma_b_cross_12:
        # buy 2 for 1
        asset_2 += delta * asset_1_prev * (1-fees) / S12
        asset_1 -= delta * asset_1_prev

    MtM = S1 * asset_1 + S2 * asset_2
    print(MtM)

    MtM_list.append(MtM)
    asset1_list.append(asset_1)
    asset2_list.append(asset_2)
    index1_list.append(initial_asset1_eq * S1)
    index2_list.append(initial_asset2_eq * S2)
    index12_list.append(initial_asset_cross_eq1 * S1 + initial_asset_cross_eq2 * S2)


# --- Plotting ---

plt.figure()
plt.title("Indexes comparison")
plt.plot(np.array(MtM_list), label="Algo")
plt.plot(np.array(index1_list), label=f"index1: {ASSET1}")
plt.plot(np.array(index2_list), label=f"index2: {ASSET2}")
plt.plot(np.array(index12_list), label=f"index12: {ASSET1} + {ASSET2}")
plt.legend()

plt.figure()
plt.title("Performance")
plt.plot(np.array(MtM_list) - np.array(index12_list))

plt.figure()
plt.title("Asset1")
plt.plot(np.array(asset1_list))

plt.figure()
plt.title("Asset2")
plt.plot(np.array(asset2_list))
