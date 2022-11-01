import pandas as pd
from ta.trend import MACD, ADXIndicator, CCIIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volume import volume_weighted_average_price, ease_of_movement
from ta.volatility import BollingerBands

"""
    This files aims to provide a set of function to compute technical indicators for given datasets
"""


# --- ONE SHOT FUNCTIONS ---

# - Trend -
def compute_MACD(df: pd.DataFrame):
    # df is supposed to contain a "Close" column
    macd = MACD(close=df.Close, fillna=True)
    df["MACD"] = macd.macd_diff()


def compute_ADX(df: pd.DataFrame):
    # df is supposed to contain a "High", "Low" and "Close" column
    # /!\ the function currently encounters troubles calculating ADX
    adx = ADXIndicator(high=df.High, low=df.Low, close=df.Close, fillna=True)
    df["ADX"] = adx.adx()


def compute_CCI(df: pd.DataFrame):
    # df is supposed to contain a "High", "Low" and "Close" column
    cci = CCIIndicator(high=df.High, low=df.Low, close=df.Close, fillna=True)
    df["ADX"] = cci.cci()


# - Momentum -
def compute_RSI(df: pd.DataFrame):
    # df is supposed to contain a "Close" column
    rsi = RSIIndicator(close=df.Close, fillna=True)
    df["RSI"] = rsi.rsi()


# - Volume -
def compute_VWAP(df: pd.DataFrame):
    # df is supposed to contain a "High", "Low", "Close" and "Volume" column
    vwap = volume_weighted_average_price(low=df.Low, high=df.High, close=df.Close, volume=df.Volume, fillna=True)
    df["VWAP"] = vwap

def compute_EOM(df: pd.DataFrame):
    # df is supposed to contain a "High", "Low" and "Volume" column
    eom = ease_of_movement(low=df.Low, high=df.High, volume=df.Volume, fillna=True)
    df["EOM"] = eom

# - Volatility -
def compute_BBANDS(df: pd.DataFrame):
    # df is supposed to contain a "Close" column
    bbands = BollingerBands(close=df.Close, fillna=True)
    df["BBANDS_PERCENT"] = bbands.bollinger_pband()
    df["BBANDS_RANGE"] = bbands.bollinger_wband()

# - Baselines -
def compute_baseline(df: pd.DataFrame, win1: int, win2: int, baseline_name: str):
    # df is supposed to contain a "Close" column
    ema2 = EMAIndicator(close=df.Close, window=win2, fillna=True)
    if win1 > 1:
        ema1 = EMAIndicator(close=df.Close, window=win1, fillna=True)
        df[baseline_name] = ema1.ema_indicator() - ema2.ema_indicator()
    else:
        df[baseline_name] = df.Close - ema2.ema_indicator()

def compute_EMA(df: pd.DataFrame, win: int, ema_name: str = "RESIDUAL"):
    # df is supposed to contain a "Close" column
    ema = EMAIndicator(close=df.Close, window=win, fillna=True)
    df[ema_name] = ema.ema_indicator()
