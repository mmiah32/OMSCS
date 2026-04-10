import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data
import indicators as ind
import marketsim as ms


class ManualStrategy(object):
    def __init__(self, verbose = False, impact = 0.0, commission = 0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def add_evidence(self, symbol='IBM', sd=dt.datetime(2008, 1, 1, 0, 0),
                     ed=dt.datetime(2009, 1, 1, 0, 0), sv=100000):
        pass

    def testPolicy(self, symbol='IBM', sd= dt.datetime(2009, 1, 1, 0, 0),
                   ed= dt.datetime(2010, 1, 1, 0, 0), sv=100000):
        start_date = sd - dt.timedelta(days=30)
        data = get_data([symbol], pd.date_range(start_date, ed))

        ticker = data[symbol]

        momentum = ind.momentum(ticker, 14)
        ema = ind.EMA(ticker, 12)
        bbp = ind.BBP(ticker, 20)
        rsi = ind.RSI(ticker, 14)
        stochastic = ind.stochastic(ticker, 14)

        tickers = pd.DataFrame({
            'momentum': momentum,
            'ema': ema,
            'bbp': bbp,
            'rsi': rsi,
            'stochastic': stochastic
        })

        tickers = tickers.dropna()

        buy_mask = (
            (   #- std below mean indicates oversold conditions
                tickers['bbp'] < -0.5
            )
                &
            (
                tickers['rsi'] < 30
            )
        )
        #print(buy_mask.value_counts())

        sell_mask = (
            (   #- std below mean indicates oversold conditions
                tickers['bbp'] < -0.5
            )
                &
            (
                tickers['rsi'] > 30
            )
                &
            (
                tickers['momentum'] < 0
            )
        )
        #print(sell_mask.value_counts())


        current_position = 0
        trades = pd.Series(0, index = tickers.index, dtype = float)
        for day in tickers.index:
            if buy_mask[day] and current_position != 1000:
                trade = 1000 - current_position
                trades.loc[day] = trade
                current_position = 1000
            elif sell_mask[day] and current_position != -1000:
                trade = -1000 - current_position
                trades.loc[day] = trade
                current_position = -1000
            else:
                trade = 0
                trades.loc[day] = trade

        trades = trades.to_frame(name = 'trade')
        trades = trades.loc[sd:ed]
        #print(trades)
        #print(trades.value_counts())

        return trades


    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "mmiah32"  # replace tb34 with your Georgia Tech username.

    def study_group(self):
        return "mmiah32", "discord groupchat"

if __name__ == '__main__':
    instance = ManualStrategy()
    result = instance.testPolicy(symbol='JPM', sd= dt.datetime(2010, 1, 1, 0, 0),
                   ed= dt.datetime(2011, 12, 31, 0, 0), sv=100000)