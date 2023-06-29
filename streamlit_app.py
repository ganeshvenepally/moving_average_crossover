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

    short_window_col_sma = str(short_window) + '_' + 'SMA'
    long_window_col_sma = str(long_window) + '_' + 'SMA'
    short_window_col_ema = str(short_window) + '_' + 'EMA'
    long_window_col_ema = str(long_window) + '_' + 'EMA'

    # Calculating SMA and EMA
    stock_df[short_window_col_sma] = stock_df['Close Price'].rolling(window = short_window, min_periods = 1).mean()
    stock_df[long_window_col_sma] = stock_df['Close Price'].rolling(window = long_window, min_periods = 1).mean()
    stock_df[short_window_col_ema] = stock_df['Close Price'].ewm(span = short_window, adjust = False).mean()
    stock_df[long_window_col_ema] = stock_df['Close Price'].ewm(span = long_window, adjust = False).mean()

    if moving_avg == 'SMA':
        stock_df['Signal'] = np.where(stock_df[short_window_col_sma] > stock_df[long_window_col_sma], 1.0, 0.0)
    elif moving_avg == 'EMA':
        stock_df['Signal'] = np.where(stock_df[short_window_col_ema] > stock_df[long_window_col_ema], 1.0, 0.0)
    elif moving_avg == 'Both':
        stock_df['Signal'] = np.where((stock_df[short_window_col_sma] > stock_df[long_window_col_sma]) & (stock_df[short_window_col_ema] > stock_df[long_window_col_ema]), 1.0, 0.0)

    stock_df['Position'] = stock_df['Signal'].diff()

    # Simulate the trading
    cash_balance = initial_cash
    stock_qty = 0
    buy_price = 0
    stock_df['Shares'] = 0
    stock_df['Cash'] = initial_cash
    stock_df['Return'] = 0
    for i in range(1, len(stock_df)):
        # Buy
        if stock_df['Position'].iloc[i] == 1 and cash_balance > 0:
            stock_qty = cash_balance // stock_df['Close Price'].iloc[i]
            buy_price = stock_df['Close Price'].iloc[i]
            cash_balance %= stock_df['Close Price'].iloc[i]
        # Sell
        elif stock_df['Position'].iloc[i] == -1 and stock_qty > 0:
            cash_balance += stock_qty * stock_df['Close Price'].iloc[i]
            returns = ((stock_df['Close Price'].iloc[i] - buy_price) / buy_price) * 100
            stock_df.loc[stock_df.index[i], 'Return'] = returns
            stock_qty = 0
        stock_df.loc[stock_df.index[i], 'Shares'] = stock_qty
        stock_df.loc[stock_df.index[i], 'Cash'] = cash_balance
    final_value = cash_balance + stock_qty * stock_df['Close Price'].iloc[-1]

    # Calculate portfolio returns from prices
    stock_df['Portfolio Returns'] = stock_df['Close Price'].pct_change()

    return stock_df, final_value

st.title("Moving Average Crossover Strategy Simulator")
stock_symbol = st.text_input("Stock Symbol:", 'ASIANPAINT.NS')
start_date = st.date_input("Start Date:", pd.to_datetime('2022-01-31'))
end_date = st.date_input("End Date:", pd.to_datetime('2023-06-16'))
short_window = st.slider("Short Window:", min_value=1, max_value=50, value=5, step=1)
long_window = st.slider("Long Window:", min_value=1, max_value=200, value=20, step=1)
moving_avg = st.selectbox("Moving Average Type:", ('SMA', 'EMA', 'Both'), index=0)
display_table = st.checkbox("Display Table?", value=True)
initial_cash = st.slider("Initial Cash:", min_value=10000, max_value=100000, value=50000, step=1000)

if st.button('Run Simulation'):
    results, final_value = MovingAverageCrossStrategy(stock_symbol, start_date, end_date, short_window, long_window, moving_avg, display_table, initial_cash)

    # Get only the returns and drop missing values
    returns = results['Portfolio Returns'].dropna()

    # Display quantstats report
    st.write(qs.reports.html(returns, "SPY"))

    # Counting the number of trades
    df_pos = results.loc[(results['Position'] == 1) | (results['Position'] == -1)].copy()
    num_trades = df_pos['Position'].value_counts()

    st.write(f"Number of Buy trades: {num_trades[1]}")
    st.write(f"Number of Sell trades: {num_trades[-1]}")
