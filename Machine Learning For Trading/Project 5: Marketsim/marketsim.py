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
    orders_file="./orders/orders.csv",  		  	   		 		  			  		 			     			  	 
    start_val=1000000,  		  	   		 		  			  		 			     			  	 
    commission=9.95,  		  	   		 		  			  		 			     			  	 
    impact=0.005,
):
    """  		  	   		 		  			  		 			     			  	 
    Computes the portfolio values.  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
    :param orders_file: Path of the order file or the file object  		  	   		 		  			  		 			     			  	 
    :type orders_file: str or file object  		  	   		 		  			  		 			     			  	 
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
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates = True, na_values = ['nan'])
    tickers = orders_df['Symbol'].unique()
    start_date = orders_df.index.min()
    end_date = orders_df.index.max()
    prices = get_data(tickers, pd.date_range(start_date, end_date))
    prices['Cash'] = 1.0

    trades = prices.copy(deep = True)
    trades = trades.drop('SPY', axis = 1)
    prices['Cash'] = 1.0
    trades.iloc[:] = 0

    for index, row in orders_df.iterrows():
        Symbol = row['Symbol']
        Order = row['Order']
        Shares = row['Shares']
        Date = index

        if Date in trades.index:
            price = prices.loc[Date, Symbol]

            if Order == 'BUY':
                new_price = price * (1.0 + impact)
                trades.loc[Date, Symbol] += Shares
                trades.loc[Date, 'Cash'] += -(Shares * new_price) - commission

            elif Order == 'SELL':
                new_price = price * (1.0 - impact)
                trades.loc[Date, Symbol] -= Shares
                trades.loc[Date, 'Cash'] += (Shares * new_price) - commission

    holdings = trades.cumsum()
    holdings['Cash'] += start_val
    portvals = (holdings * prices).sum(axis=1)

    return portvals


def test_code():  		  	   		 		  			  		 			     			  	 
    """  		  	   		 		  			  		 			     			  	 
    Helper function to test code  		  	   		 		  			  		 			     			  	 
    """  		  	   		 		  			  		 			     			  	 
    # this is a helper function you can use to test your code  		  	   		 		  			  		 			     			  	 
    # note that during autograding his function will not be called.  		  	   		 		  			  		 			     			  	 
    # Define input parameters

    compute_portvals("./orders/orders-01.csv")

    of = "./orders/orders2.csv"
    sv = 1000000  		  	   		 		  			  		 			     			  	 
  		  	   		 		  			  		 			     			  	 
    # Process orders  		  	   		 		  			  		 			     			  	 
    portvals = compute_portvals(orders_df=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):  		  	   		 		  			  		 			     			  	 
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 		  			  		 			     			  	 
    else:  		  	   		 		  			  		 			     			  	 
        "warning, code did not return a DataFrame"  		  	   		 		  			  		 			     			  	 


if __name__ == "__main__":  		  	   		 		  			  		 			     			  	 
    test_code()  		  	   		 		  			  		 			     			  	 
