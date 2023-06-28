import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

def MovingAverageCrossStrategy(stock_symbol, start_date, end_date, short_window, long_window, moving_avg, display_table, initial_cash):
    # Get the stock data
    stock = yf.Ticker(stock_symbol)
    stock_df = stock.history(start=start_date, end=end_date)['Close']
    stock_df = pd.DataFrame(stock_df)
    stock_df.columns = {'Close Price'}

    short_window_col = str(short_window) + '_' + moving_avg
    long_window_col = str(long_window) + '_' + moving_avg

    if moving_avg == 'SMA':
        stock_df[short_window_col] = stock_df['Close Price'].rolling(window = short_window, min_periods = 1).mean()
        stock_df[long_window_col] = stock_df['Close Price'].rolling(window = long_window, min_periods = 1).mean()
    elif moving_avg == 'EMA':
        stock_df[short_window_col] = stock_df['Close Price'].ewm(span = short_window, adjust = False).mean()
        stock_df[long_window_col] = stock_df['Close Price'].ewm(span = long_window, adjust = False).mean()

    stock_df['Signal'] = 0.0  
    stock_df['Signal'] = np.where(stock_df[short_window_col] > stock_df[long_window_col], 1.0, 0.0)
    stock_df['Position'] = stock_df['Signal'].diff()

    # Simulate the trading
    cash_balance = initial_cash
    stock_qty = 0
    stock_df['Shares'] = 0
    stock_df['Cash'] = initial_cash
    stock_df['Return'] = 0
    for i in range(1, len(stock_df)):
        # Buy
        if stock_df['Position'].iloc[i] == 1 and cash_balance > 0:
            stock_qty += cash_balance // stock_df['Close Price'].iloc[i]
            cash_balance %= stock_df['Close Price'].iloc[i]
        # Sell
        elif stock_df['Position'].iloc[i] == -1 and stock_qty > 0:
            cash_balance += stock_qty * stock_df['Close Price'].iloc[i]
            stock_qty = 0
        stock_df.loc[stock_df.index[i], 'Shares'] = stock_qty
        stock_df.loc[stock_df.index[i], 'Cash'] = cash_balance
        stock_df.loc[stock_df.index[i], 'Return'] = (cash_balance + stock_qty * stock_df['Close Price'].iloc[i]) / initial_cash - 1
    final_value = cash_balance + stock_qty * stock_df['Close Price'].iloc[-1]
    st.write(f'Final portfolio value: {final_value}')

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(stock_df['Close Price'], color='k', lw=1, label='Close Price')
    ax.plot(stock_df[short_window_col], color='b', lw=1, label=short_window_col)
    ax.plot(stock_df[long_window_col], color='g', lw=1, label=long_window_col)
    ax.plot(stock_df[stock_df['Position'] == 1].index, stock_df[short_window_col][stock_df['Position'] == 1], '^', markersize=15, color='g', alpha=0.7, label='buy')
    ax.plot(stock_df[stock_df['Position'] == -1].index, stock_df[short_window_col][stock_df['Position'] == -1], 'v', markersize=15, color='r', alpha=0.7, label='sell')
    plt.ylabel('Price in ₹', fontsize=16)
    plt.xlabel('Date', fontsize=16)
    plt.title(str(stock_symbol) + ' - ' + str(moving_avg) + ' Crossover', fontsize=20)
    plt.legend()
    plt.grid()
    st.pyplot(fig)

    if display_table:
        df_pos = stock_df.loc[(stock_df['Position'] == 1) | (stock_df['Position'] == -1)].copy()
        df_pos['Position'] = df_pos['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
        st.table(df_pos)  # Displaying the table

# UI Elements
st.title("Moving Average Crossover Strategy Simulator")
st.write('Please input the required information to run the Moving Average Crossover Strategy Simulator.')

# Inputs
stock_symbol = st.text_input("Stock Symbol:", 'ASIANPAINT.NS')
start_date = st.date_input("Start Date:", pd.to_datetime('2022-01-31'))
end_date = st.date_input("End Date:", pd.to_datetime('2023-06-16'))
short_window = st.slider("Short Window:", min_value=1, max_value=50, value=5, step=1)
long_window = st.slider("Long Window:", min_value=1, max_value=200, value=20, step=1)
moving_avg = st.selectbox("Moving Average Type:", ('SMA', 'EMA'), index=0)
display_table = st.checkbox("Display Table?", value=True)
initial_cash = st.slider("Initial Cash:", min_value=10000, max_value=100000, value=50000, step=1000)

# Button to start the function
if st.button('Run Simulation'):
    MovingAverageCrossStrategy(stock_symbol, start_date, end_date, short_window, long_window, moving_avg, display_table, initial_cash)
