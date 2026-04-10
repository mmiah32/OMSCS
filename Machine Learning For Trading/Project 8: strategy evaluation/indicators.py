import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data
import datetime as dt

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "mmiah32"  # replace tb34 with your Georgia Tech username.

def study_group():
    return "mmiah32", "discord groupchat"

def momentum(price, N):
    momentum = (price / price.shift(N)) - 1
    return momentum


def EMA(price, N):
    ema = price.ewm(span=N).mean()
    ema_signal = (price / ema) - 1
    return ema_signal


def BBP(price, N):
    std = price.rolling(N).std()
    sma = price.rolling(N).mean()
    bb_value = (price - sma) / (2 * std)
    return bb_value


def RSI(price, N):
    daily_price_change = (price - price.shift(1))
    gains = daily_price_change.clip(lower = 0)

    #multiply by -1 keeps values positive
    losses = daily_price_change.clip(upper = 0) * -1
    avg_gains = gains.rolling(N).mean()
    avg_losses = losses.rolling(N).mean()

    rs = avg_gains / avg_losses
    rsi = 100 - 100 / (1 + rs)

    return rsi


def stochastic(price, N):
    highest_high = price.rolling(N).max()
    lowest_low = price.rolling(N).min()
    stochastic = (price - lowest_low) / (highest_high - lowest_low) * 100
    return stochastic


def z_score(indicator):
    z_score = (indicator - indicator.mean()) / indicator.std()
    return z_score




