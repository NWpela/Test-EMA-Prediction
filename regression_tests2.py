import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import indicators as ind
from scipy import signal
from sklearn.linear_model import LinearRegression

matplotlib.use('Qt5Agg')

"""
    This file aims to plot the different baselines for the baseline model
"""

# --- Parameters Definition ---

ASSET = "BTC"
FIELD = "Close"
N_TRAIN = 30000
N_TEST = 15000

EWA_WINDOWS_LIST = [(1, 10)]
WITH_RESIDUAL = True


# --- Data Loading, Indicators ,Normalisation and baselines---

raw_df = pd.read_csv(f"binance_data/{ASSET}USDT_1m_v1.csv", sep=';')[:(N_TRAIN+N_TEST)]
raw_df_down = pd.read_csv(f"binance_data/{ASSET}DOWNUSDT_1m_v1.csv", sep=';')[:(N_TRAIN+N_TEST)]

for ewa_windows in EWA_WINDOWS_LIST:
    baseline_name = f"BASELINE_{ewa_windows[0]}_{ewa_windows[1]}"
    ind.compute_baseline(raw_df, ewa_windows[0], ewa_windows[1], baseline_name)
# compute residual if needed
if WITH_RESIDUAL:
    ind.compute_EMA(raw_df, EWA_WINDOWS_LIST[-1][1])

ind.compute_RSI(raw_df)
ind.compute_VWAP(raw_df)

raw_df["NEXT_CLOSE_MOVE"] = raw_df["Close"].diff().shift(periods=-1)
raw_df["VWAP_CLOSE_DIFF"] = raw_df["VWAP"] - raw_df["Close"]
raw_df["VWAP_DIFF"] = raw_df["VWAP"].diff()
raw_df["RSI_DIFF"] = raw_df["RSI"].diff()
raw_df["CLOSE_MOVE"] = raw_df["Close"].diff()


# --- Regressions ---

USED_FIELD = "VWAP_DIFF"
x = np.array(raw_df[USED_FIELD].iloc[1:(N_TRAIN-1)]).reshape(-1, 1)
y = np.array(raw_df["NEXT_CLOSE_MOVE"].iloc[1:(N_TRAIN-1)]).reshape(-1, 1)

regr = LinearRegression()
regr.fit(x, y)
print(regr.score(x, y))
y_pred = regr.predict(x)

plt.figure()
plt.scatter(x, y, color='b', marker='+')
plt.plot(x, y_pred, color='k')


# --- Backtesting ---

cash = 100
delta = 0.1

btc = 0

cash_list = []
MtM_list = []
for k in range(1, N_TEST-1):
    data = raw_df.iloc[N_TRAIN + k]
    data_down = raw_df_down.iloc[N_TRAIN + k]

    x = data[USED_FIELD]
    predicted_close_var = regr.predict(np.array([[x]]))
    close = data.Close
    close_down = data_down.Close

    # save cash value
    cash_list.append(cash)
    MtM_list.append(cash + close * btc)

    # buy next ones
    if predicted_close_var < 0:  # sign is strange here
        btc += delta * cash / close
        cash -= delta * cash
    else:
        cash += delta * btc * close
        btc -= delta * btc

plt.figure()
plt.plot(np.array(MtM_list))
