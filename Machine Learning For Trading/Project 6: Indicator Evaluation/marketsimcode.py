""""""
"""MC2-P1: Market simulator.  		  	   		 		  			  		 			     			  	 

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
GT ID: 904188350 (replace with your GT ID)  		  	   		 		  			  		 			     			  	 
"""

import datetime as dt
import os

import numpy as np

import pandas as pd
from util import get_data, plot_data


def author():
    return 'mmiah32'


def study_group():
    return "mmiah32", "discord groupchat"


def compute_portvals(
        trades_df,
        symbol,
        start_val=100000
):
    """
    Computes the portfolio values.

    :param trades_df: Path of the order file or the file object
    :type trades_df: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here
    start_date = trades_df.index.min()
    end_date = trades_df.index.max()
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices['Cash'] = 1.0


    trades = prices.copy(deep=True)
    trades = trades.drop('SPY', axis=1)
    trades.iloc[:] = 0

    for index, row in trades_df.iterrows():
        trade = row['trades']
        date = index

        if date in prices.index:
            price = prices.loc[date, symbol]
            trades.loc[date, symbol] += trade
            trades.loc[date, 'Cash'] += -(trade * price)

    holdings = trades.cumsum()
    holdings['Cash'] += start_val
    portvals = (holdings * prices).sum(axis=1)

    return portvals


if __name__ == "__main__":
    pass