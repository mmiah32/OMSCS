""""""
"""  		  	   		 		  			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 		  			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 		  			  		 			     			  	 
All Rights Reserved  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 		  			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 		  			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 		  			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 		  			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 		  			  		 			     			  	 
or edited.  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 		  			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 		  			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 		  			  		 			     			  	 
GT honor code violation.  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
Student Name: Mohammed Miah (replace with your name)  		  	   		 		  			  		 			     			  	 
GT User ID: mmiah32 (replace with your User ID)  		  	   		 		  			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		 		  			  		 			     			  	 
"""  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
import datetime as dt  		  	   		 		  			  		 			     			  	 
import random  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
import pandas as pd  		  	   		 		  			  		 			     			  	 
import util as ut
import numpy as np
import datetime as dt
from util import get_data
import indicators as ind
import BagLearner as bl
import RTLearner as rt
  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
class StrategyLearner(object):  		  	   		 		  			  		 			     			  	 
    """  		  	   		 		  			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 		  			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 		  			  		 			     			  	 
    :type verbose: bool  		  	   		 		  			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 		  			  		 			     			  	 
    :type impact: float  		  	   		 		  			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 		  			  		 			     			  	 
    :type commission: float  		  	   		 		  			  		 			     			  	 
    """

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "mmiah32"  # replace tb34 with your Georgia Tech username.

    def study_group(self):
        return "mmiah32", "discord groupchat"

    # constructor  		  	   		 		  			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		 		  			  		 			     			  	 
        Constructor method  		  	   		 		  			  		 			     			  	 
        """  		  	   		 		  			  		 			     			  	 
        self.verbose = verbose  		  	   		 		  			  		 			     			  	 
        self.impact = impact  		  	   		 		  			  		 			     			  	 
        self.commission = commission
        self.learner = bl.BagLearner(rt.RTLearner, {'leaf_size': 10}, 20, boost = False, verbose = False)

  		  	   		 		  			  		 			     			  	 
    # this method should create a QLearner, and train it for trading  		  	   		 		  			  		 			     			  	 
    def add_evidence(  		  	   		 		  			  		 			     			  	 
        self,  		  	   		 		  			  		 			     			  	 
        symbol="IBM",  		  	   		 		  			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		 		  			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),  		  	   		 		  			  		 			     			  	 
        sv=10000,  		  	   		 		  			  		 			     			  	 
    ):  		  	   		 		  			  		 			     			  	 
        """  		  	   		 		  			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		 		  			  		 			     			  	 
        :type symbol: str  		  	   		 		  			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 		  			  		 			     			  	 
        :type sd: datetime  		  	   		 		  			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 		  			  		 			     			  	 
        :type ed: datetime  		  	   		 		  			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 		  			  		 			     			  	 
        :type sv: int  		  	   		 		  			  		 			     			  	 
        """  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
        # add your code to do learning here
        #warmup period
        start_date = sd - dt.timedelta(days=30)
        data = get_data([symbol], pd.date_range(start_date, ed))

        #drop spy
        ticker = data[symbol]

        momentum_z = ind.z_score(ind.momentum(ticker, 14))
        ema_z = ind.z_score(ind.EMA(ticker, 12))
        bbp_z = ind.z_score(ind.BBP(ticker, 20))
        rsi_z = ind.z_score(ind.RSI(ticker, 14))
        stochastic_z = ind.z_score(ind.stochastic(ticker, 14))

        indicators_z = pd.DataFrame({
            'momentum': momentum_z,
            'ema': ema_z,
            'bbp': bbp_z,
            'rsi': rsi_z,
            'stochastic': stochastic_z
        })

        future_return = (ticker.shift(-5) / ticker) - 1

        #create label (1) where future_return > in 5
        indicators_z['labels'] = np.where(future_return > 0.005 + self.impact * 2, 1, 0)
        #create label (-1) where future_return < -0.005
        #values < 0.005 and > -0.005 default to 0
        indicators_z['labels'] = np.where(future_return < -(0.005 + self.impact * 2), -1, indicators_z['labels'])

        #drop NA values from warmup period and
        # last 5 days since no 6th day to check returns
        indicators_z = indicators_z.dropna()
        indicators_z = indicators_z.loc[sd:ed]

        #x values
        x_values = indicators_z[['momentum', 'ema', 'bbp', 'rsi', 'stochastic']]
        y_values = indicators_z['labels']

        x_values = np.array(x_values)
        y_values = np.array(y_values)

        self.learner.add_evidence(x_values, y_values)


    # this method should use the existing policy and test it against new data  		  	   		 		  			  		 			     			  	 
    def testPolicy(  		  	   		 		  			  		 			     			  	 
        self,  		  	   		 		  			  		 			     			  	 
        symbol="IBM",  		  	   		 		  			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),  		  	   		 		  			  		 			     			  	 
        ed=dt.datetime(2010, 1, 1),  		  	   		 		  			  		 			     			  	 
        sv=10000,  		  	   		 		  			  		 			     			  	 
    ):  		  	   		 		  			  		 			     			  	 
        """  		  	   		 		  			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 		  			  		 			     			  	 
        :type symbol: str  		  	   		 		  			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 		  			  		 			     			  	 
        :type sd: datetime  		  	   		 		  			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 		  			  		 			     			  	 
        :type ed: datetime  		  	   		 		  			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 		  			  		 			     			  	 
        :type sv: int  		  	   		 		  			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 		  			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 		  			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 		  			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 		  			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 		  			  		 			     			  	 
        """

        # add your code to do learning here
        #warmup period
        start_date = sd - dt.timedelta(days=30)
        data = get_data([symbol], pd.date_range(start_date, ed))

        #drop spy
        ticker = data[symbol]

        momentum_z = ind.z_score(ind.momentum(ticker, 14))
        ema_z = ind.z_score(ind.EMA(ticker, 12))
        bbp_z = ind.z_score(ind.BBP(ticker, 20))
        rsi_z = ind.z_score(ind.RSI(ticker, 14))
        stochastic_z = ind.z_score(ind.stochastic(ticker, 14))

        indicators_z = pd.DataFrame({
            'momentum': momentum_z,
            'ema': ema_z,
            'bbp': bbp_z,
            'rsi': rsi_z,
            'stochastic': stochastic_z
        })

        indicators_z = indicators_z.dropna()
        indicators_z = indicators_z.loc[sd:ed]
        indicators_z_index = indicators_z.index

        indicators_z = np.array(indicators_z)

        result = self.learner.query(indicators_z)

        #create trades df
        current_position = 0
        trades = pd.Series(0, index = indicators_z_index, dtype = float)
        for i, value in enumerate(result):
            i = indicators_z_index[i]

            if value == 1 and current_position != 1000:
                trade = 1000 - current_position
                trades.loc[i] = trade
                current_position = 1000
            elif value == -1 and current_position != -1000:
                trade = -1000 - current_position
                trades.loc[i] = trade
                current_position = -1000
            else:
                trade = 0
                trades.loc[i] = trade

        trades = trades.to_frame(name = 'trade')
        #print(trades.value_counts())

        return trades


if __name__ == "__main__":
    instance = StrategyLearner()

    instance.add_evidence(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=10000
    )

    print(instance.testPolicy(
        symbol="JPM",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=10000
    ))

