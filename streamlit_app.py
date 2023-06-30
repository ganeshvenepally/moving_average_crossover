import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import quantstats as qs

def MovingAverageCrossStrategy(stock_symbol, start_date, end_date, short_window, long_window, moving_avg, display_table, initial_cash):
    # Get the stock data
    stock = yf.Ticker(stock_symbol)
    stock_df = stock.history(start=start_date, end=end_date)['Close']
    stock_df = pd.DataFrame(stock_df)
    stock_df.columns = {'Close Price'}

    short_window_col_sma = str(short_window) + '_SMA'
    long_window_col_sma = str(long_window) + '_SMA'
    short_window_col_ema = str(short_window) + '_EMA'
    long_window_col_ema = str(long_window) + '_EMA'
    
    # Calculating SMA
    stock_df[short_window_col_sma] = stock_df['Close Price'].rolling(window = short_window, min_periods = 1).mean()
    stock_df[long_window_col_sma] = stock_df['Close Price'].rolling(window = long_window, min_periods = 1).mean()
    
    # Calculating EMA
    stock_df[short_window_col_ema] = stock_df['Close Price'].ewm(span = short_window, adjust = False).mean()
    stock_df[long_window_col_ema] = stock_df['Close Price'].ewm(span = long_window, adjust = False).mean()

    if moving_avg == 'SMA':
        stock_df['Signal'] = np.where(stock_df[short_window_col_sma] > stock_df[long_window_col_sma], 1.0, 0.0)
    elif moving_avg == 'EMA':
        stock_df['Signal'] = np.where(stock_df[short_window_col_ema] > stock_df[long_window_col_ema], 1.0, 0.0)
    elif moving_avg == 'Both':
        stock_df['Signal'] = np.where((stock_df[short_window_col_sma] > stock_df[long_window_col_sma]) & (stock_df[short_window_col_ema] > stock_df[long_window_col_ema]), 1.0, 0.0)
        
    stock_df['Position'] = stock_df['Signal'].diff()
    
    # Create portfolio to store asset data
    portfolio = pd.DataFrame(index=stock_df.index, data={'Asset': stock_df['Close Price']})
    
    # Buy/Sell signals
    signals = pd.concat([
        pd.DataFrame({"Price": stock_df.loc[stock_df["Position"] == 1, "Close Price"],
                      "Regime": stock_df.loc[stock_df["Position"] == 1, "Signal"],
                      "Signal": "Buy"}),
        pd.DataFrame({"Price": stock_df.loc[stock_df["Position"] == -1, "Close Price"],
                      "Regime": stock_df.loc[stock_df["Position"] == -1, "Signal"],
                      "Signal": "Sell"}),
    ])
    signals.sort_index(inplace=True)
    
    # Add signals to portfolio
    portfolio = portfolio.join(signals[["Signal"]], how="outer").fillna(value={"Signal": "Wait"})
    
    # Add "Shares" column to portfolio
    portfolio["Shares"] = [initial_cash/portfolio.loc[row, "Asset"] if portfolio.loc[row, "Signal"] == "Buy" else 0 for row in portfolio.index]
    
    # Add "Cash" column to portfolio
    portfolio['Cash'] = initial_cash - (portfolio['Shares'] * portfolio['Asset'])
    
    # Add "Total" column to portfolio
    portfolio['Total'] = portfolio['Shares'] * portfolio['Asset'] + portfolio['Cash']
    
    return portfolio, stock_df, signals

# User inputs
stock_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-06-30'
short_window = 50
long_window = 200
moving_avg = 'Both'
display_table = True
initial_cash = 100000.0

portfolio, stock_df, signals = MovingAverageCrossStrategy(stock_symbol, start_date, end_date, short_window, long_window, moving_avg, display_table, initial_cash)

# Create quantstats report
returns = portfolio['Total'].pct_change().iloc[1:]
qs.reports.html(returns, output='quantstats.html')

# Create plot
stock_df['Close Price'].plot(color = 'blue', label= 'Close Price')
stock_df[str(short_window)+'_SMA'].plot(color = 'red', label = 'Short-window SMA')
stock_df[str(long_window)+'_SMA'].plot(color = 'green', label = 'Long-window SMA')
stock_df[str(short_window)+'_EMA'].plot(color = 'orange', label = 'Short-window EMA')
stock_df[str(long_window)+'_EMA'].plot(color = 'magenta', label = 'Long-window EMA')
plt.legend()
plt.show()

# Show the transaction table
if display_table:
    print(signals)
