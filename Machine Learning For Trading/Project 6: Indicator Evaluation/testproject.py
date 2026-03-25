import TheoreticallyOptimalStrategy as tos
import marketsimcode as msc
import indicators as ind
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

def run_tos_analysis(symbol = "JPM", sd= dt.datetime(2008, 1, 1), ed= dt.datetime(2009,12,31), sv = 100000):
    prices = get_data([symbol], pd.date_range(sd, ed))
    jpm_price = prices[symbol]
    create_charts(jpm_price, n_momentum=14, n_bbp=20, n_ema=12, n_rsi=14, n_stochastic=14)

    df_trades = tos.testPolicy(symbol, sd, ed, sv)
    tos_portvals = msc.compute_portvals(df_trades, symbol, start_val = sv)

    benchmark = pd.DataFrame(
        {"trades": 0},
        index=tos_portvals.index
    )
    benchmark.iloc[0] = 1000
    benchmark_values = msc.compute_portvals(benchmark, symbol, start_val = sv)
    normalized_benchmark = benchmark_values.iloc[:] / benchmark_values.iloc[0]

    normalized_tos_portvals = tos_portvals[:] / tos_portvals.iloc[0]

    plt.figure()
    plt.plot(normalized_tos_portvals, label = "Normalized TOS Trades", color = "red")
    plt.plot(normalized_benchmark, label = 'Normalized Benchmark trades', color = "purple")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Values")
    plt.title("JPM: Perfect Foresight Trading (TOS) \n vs \n Buy-and-Hold Benchmark")
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig('./images/fig1.png')
    #plt.show()

    cr_tos = ((tos_portvals.iloc[-1] / tos_portvals.iloc[0]) - 1).round(6)
    daily_returns_tos = (tos_portvals / tos_portvals.shift(1)) - 1
    adr_tos = round(daily_returns_tos.mean(), 6)
    sdr_tos = round(daily_returns_tos.std(), 6)

    cr_benchmark = ((benchmark_values.iloc[-1] / benchmark_values.iloc[0]) - 1).round(6)
    daily_returns_benchmark = (benchmark_values / benchmark_values.shift(1)) - 1
    adr_benchmark = round(daily_returns_benchmark.mean(), 6)
    sdr_benchmark = round(daily_returns_benchmark.std(), 6)

    stats = pd.DataFrame({
        'TOS': [cr_tos, adr_tos, sdr_tos],
        'Benchmark': [cr_benchmark, adr_benchmark, sdr_benchmark]
    }, index=['Cumulative Return', 'Mean Daily Return', 'Stdev Daily Return'])


def create_charts(prices, n_momentum=14, n_bbp=20, n_ema=12, n_rsi=14, n_stochastic=14):
    #normalize prices by dividing all prices by value in first row
    normalized_prices = prices / prices.iloc[0]

    #MOMENTUM
    #calculate momentum as % of price change over N days (default 14 days)
    raw_momentum = (prices / prices.shift(n_momentum)) - 1

    # 2 rows, 1 column of plots --> 2 graphs on top of each other
    # ax1 is first graph, ax2 is second graph
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    #plot normalized prices on first chart
    ax1.plot(normalized_prices, label="Normalized Price", color="red")
    ax1.set_ylabel("Normalized Price")
    ax1.set_title("JPM Prices")
    #set legend to upper right of first chart
    ax1.legend(loc="upper right")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    #plot momentum values on second chart
    ax2.plot(raw_momentum, label="Momentum", color="darkorchid")
    #set neutral line at 0
    ax2.axhline(y=0, color="black", linestyle="--", label="Zero Line")
    #set bullish threshold at 0.1 (+10%) reduces noise at small fluctuations
    #above 0
    ax2.axhline(y=0.1, color="green", linestyle=":", label="Bullish Threshold (0.1)")
    #set bearish threshold at -0.1 (-0.10) reduces noise at small fluctuations below
    #0
    ax2.axhline(y=-0.1, color="red", linestyle=":", label="Bearish Threshold (-0.1)")

    # Shade green above zero to represent positive momentum (bullish)
    ax2.fill_between(raw_momentum.index, raw_momentum, 0,
                     where=(raw_momentum > 0),
                     color='lightgreen', alpha=0.3, label="Positive Momentum")
    # share red below zero to represent negative momentum (bearish)
    ax2.fill_between(raw_momentum.index, raw_momentum, 0,
                     where=(raw_momentum < 0),
                     color='lightcoral', alpha=0.3, label="Negative Momentum")

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Momentum")
    ax2.set_title("JPM: 14 Day Momentum Signal")
    ax2.legend(loc="upper right")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./images/momentum.png')
    #plt.show()

    # EMA
    #calculate ema values over default window size using mean
    #applies exponentially decaying weights where
    #recent prices get higher weight influencing
    #avg more than older prices
    raw_ema = prices.ewm(span=n_ema).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    #plot normalized prices on first chart
    ax1.plot(normalized_prices, label="Normalized Price", color="red")
    ax1.set_ylabel("Normalized Price")
    ax1.set_title("JPM Prices")
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    #plot how far above or below today's price is relative to EMA
    #as a %
    ax2.plot(prices / raw_ema - 1, label="Price/EMA - 1", color="teal")
    #plot zero line (neutral line)
    ax2.axhline(y=0, color="black", linestyle="--", label="Zero Line")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price / EMA - 1")
    ax2.set_title("JPM: Price Distance from 12-Day EMA")
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('./images/ema.png')
    #plt.show()

    # BBP
    #calculate sma using default window size using
    #rolling mean. This is 'middle band' of BBP
    sma = prices.rolling(n_bbp).mean()
    # calculate std using default window size using
    # rolling std. Captures volatility around rolling mean
    std = prices.rolling(n_bbp).std()
    #calculates BB% where values btwn 0 and 1.
    #0 means price at lower band. 1 means price at upper
    #band. 0.5 at midline (sma)
    raw_bbp = (prices - (sma - 2 * std)) / (4 * std)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    #plot normalized prices
    ax1.plot(normalized_prices, label="Normalized Price", color="red")
    ax1.set_ylabel("Normalized Price")
    ax1.set_title("JPM Prices")
    ax1.legend(loc="upper right")
    ax1.tick_params(axis='x', rotation=45)

    #plot BB values as %
    ax2.plot(raw_bbp, label="BB Percent", color="mediumslateblue")
    #sets upperbound of BBA to 1.0
    ax2.axhline(y=1.0, color="tomato", linestyle="--", label="Upper Threshold (1.0)")
    #sets middle of bolinger band at 0.5 (sma)
    ax2.axhline(y=0.5, color="mediumpurple", linestyle="--", label="Midline (0.5)")
    #sets lower bound of bollinger band at 0.0
    ax2.axhline(y=0.0, color="mediumseagreen", linestyle="--", label="Lower Threshold (0.0)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Bollinger Bands %")
    ax2.set_title("JPM prices: 20 Day Bollinger Bands")
    ax2.legend(loc="upper right")
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('./images/bbp.png')
    #plt.show()

    # RSI
    #creates daily change in value by subtracting today's value from yesterday's value
    daily_change = prices - prices.shift(1)
    #isolates gains where values >0
    gains = daily_change.clip(lower=0)
    #isolates losses where values <0.
    #multiply by -1 to flip to positive
    #since required for RSI
    losses = daily_change.clip(upper=0) * -1
    #calculate avg gain over default window period
    #which smooths out graph where spike in gains
    avg_gains = gains.rolling(n_rsi).mean()
    #calculate avg losses over default window period
    #which smooths out graph where spike in losses
    #keeps it positive
    avg_losses = losses.rolling(n_rsi).mean()
    #ratio of avg gains to losses
    #high rs: gains are stronger than losses on avg
    #low rs: losses are higher than gains on avg
    rs = avg_gains / avg_losses
    #Apply rsi to normalize rs to 0-100 scale
    #rsi approaches 100 when gains dominate
    #rsi approaches 0 when losses dominate
    raw_rsi = 100 - (100 / (1 + rs))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(normalized_prices, label="Normalized Price", color="red")
    ax1.set_ylabel("Normalized Price")
    ax1.set_title("JPM Prices")
    ax1.legend(loc="upper right")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    #plot raw rsi values
    ax2.plot(raw_rsi, label="RSI", color="darkorange")
    #create overbought threshold at 70 because of J. Welles Wilder
    ax2.axhline(y=70, color="darkviolet", linestyle="--", label="Overbought (70)")
    # create midline threshold at 50 — point where average gains equal average
    # losses (rs = 1)
    ax2.axhline(y=50, color="deepskyblue", linestyle="--", label="Midline (50)")
    #create oversold threshold at 30 beacuse of J. Welles Wilder
    ax2.axhline(y=30, color="limegreen", linestyle="--", label="Oversold (30)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("RSI Value")
    ax2.set_title("JPM: 14 Day RSI Signal")
    ax2.legend(loc="upper right")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./images/rsi.png')
    #plt.show()

    # STOCHASTIC
    #find maximum price over time using default window period
    highest_high = prices.rolling(n_stochastic).max()
    #find minimum price over time using default window period
    lowest_low = prices.rolling(n_stochastic).min()
    #measures where todays price sits in recent high low channel
    #100 means price is at top of recent range
    #0 means price is at bottom of recent range
    #50 means price is exactly at middle range (neutral)
    raw_stochastic = (prices - lowest_low) / (highest_high - lowest_low) * 100
    #smooths out raw stochastic value by applying rolling mean of window size
    #of 3
    smoothed_stochastic = raw_stochastic.rolling(3).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    #plot normalized prices
    ax1.plot(normalized_prices, label="Normalized Price", color="red")
    ax1.set_ylabel("Normalized Price")
    ax1.set_title("JPM Prices")
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    #plot smoothed stochastic
    ax2.plot(smoothed_stochastic, label="Stochastic %K (smoothed)", color="seagreen")
    #create overbought threshold at 80
    ax2.axhline(y=80, color="crimson", linestyle="--", label="Overbought (80)")
    #create midline at 50
    ax2.axhline(y=50, color="goldenrod", linestyle="--", label="Midline (50)")
    #create oversold threshold at 20
    ax2.axhline(y=20, color="steelblue", linestyle="--", label="Oversold (20)")

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Stochastic Value")
    ax2.set_title("JPM: 14 Day Stochastic Signal")
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('./images/stochastic.png')
    #plt.show()


if __name__ == "__main__":
    run_tos_analysis()








