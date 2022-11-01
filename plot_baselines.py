import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from indicators import compute_baseline, compute_EMA
from scipy import signal

matplotlib.use('Qt5Agg')

"""
    This file aims to plot the different baselines for the baseline model
"""

# --- Parameters Definition ---

ASSET = "BTC"
FIELD = "Close"
N_TOT = 3000

EWA_WINDOWS_LIST = [(1, 10), (10, 70), (70, 200)]
#EWA_WINDOWS_LIST = [(1, 10)]
WITH_RESIDUAL = True


# --- Data Loading, Indicators ,Normalisation and baselines---

raw_df = pd.read_csv(f"binance_data/{ASSET}EUR_15m_v1.csv", sep=';')[:N_TOT]

for ewa_windows in EWA_WINDOWS_LIST:
    baseline_name = f"BASELINE_{ewa_windows[0]}_{ewa_windows[1]}"
    compute_baseline(raw_df, ewa_windows[0], ewa_windows[1], baseline_name)
# compute residual if needed
if WITH_RESIDUAL:
    compute_EMA(raw_df, EWA_WINDOWS_LIST[-1][1])


# --- Plotting ---

plt.figure()
N_plots = len(EWA_WINDOWS_LIST) + int(WITH_RESIDUAL) + 1
fig, ax = plt.subplots(nrows=N_plots)

for k in range(N_plots - 1):
    if k == N_plots - 2 and WITH_RESIDUAL:
        ax[k].plot(np.array(raw_df["RESIDUAL"]))
        ax[k].set_title("RESIDUAL")
    else:
        baseline_name = f"BASELINE_{EWA_WINDOWS_LIST[k][0]}_{EWA_WINDOWS_LIST[k][1]}"
        ax[k].plot(np.array(raw_df[baseline_name]))
        ax[k].set_title(baseline_name)
ax[N_plots-1].plot(np.array(raw_df["Close"]))
ax[N_plots-1].set_title("Prices")


# --- Baseline Analysis (draft) ---

#plt.figure()
#bline_name = "BASELINE_1_10"
#plt.hist(np.array(raw_df[bline_name]), bins=100)

# correlations
#plt.figure()
#bline_name1, bline_name2 = "BASELINE_1_10", "BASELINE_10_70"
#bline1, bline2 = np.array(raw_df[bline_name1]), np.array(raw_df[bline_name2])
#corr = signal.correlate(bline1, bline2)
#plt.plot(corr)

# fft
#plt.figure()
#bline_fft_name = "BASELINE_10_70"
#fft = np.fft.fft(np.array(raw_df[bline_fft_name]))
#plt.plot(fft)


# --- Regression tests ---

#plt.figure()
#raw_df["CLOSE_MOVE"] = raw_df["Close"].diff().shift(periods=-1)
#bline_data = np.array(raw_df["BASELINE_1_10"].iloc[:(N_TOT-1)])
#move_data = np.array(raw_df["CLOSE_MOVE"].iloc[:(N_TOT-1)])
#pol = np.polyfit(bline_data, move_data, deg=1)
#x_min, x_max = min(bline_data), max(bline_data)
#x_line = np.linspace(x_min, x_max, 100)
#y_line = np.polyval(pol, x_line)
#plt.scatter(bline_data, move_data, marker='+')
#plt.plot(x_line, y_line, color='red')

