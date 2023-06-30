import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import time


def run_simulations(stock_symbol, start_date, end_date, short_window, long_window, display_table, initial_cash):
    #moving_avgs = ['SMA', 'EMA', 'Both']
    moving_avgs = ['SMA']
    results = []
    for moving_avg in moving_avgs:
        result = MovingAverageCrossStrategy(stock_symbol, start_date, end_date, short_window, long_window, moving_avg, display_table, initial_cash)
        results.append(result)
        st.subheader(f"Results for {moving_avg}")
        st.write(f"Final Portfolio Value (Strategy): {result[1]}")
        st.write(f"Return % (Strategy): {result[2]}%")
        st.write(f"Final Portfolio Value (Buy and Hold): {result[3]}")
        st.write(f"Return % (Buy and Hold): {result[4]}%")
        st.write(f"Number of Buy Trades: {result[5]}")
        st.write(f"Number of Sell Trades: {result[6]}")
    results_df = pd.DataFrame(results, columns=['Moving Average', 'Final Portfolio Value (Strategy)', 'Return % (Strategy)', 'Final Portfolio Value (Buy and Hold)', 'Return % (Buy and Hold)', 'Number of Buy Trades', 'Number of Sell Trades', 'Maximum Drawdown'])
    st.subheader("Summary Table for all Moving Averages")
    st.dataframe(results_df)

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

    stock_df[short_window_col_sma] = stock_df['Close Price'].rolling(window = short_window, min_periods = 1).mean()
    stock_df[long_window_col_sma] = stock_df['Close Price'].rolling(window = long_window, min_periods = 1).mean()
    stock_df[short_window_col_ema] = stock_df['Close Price'].ewm(span = short_window, adjust = False).mean()
    stock_df[long_window_col_ema] = stock_df['Close Price'].ewm(span = long_window, adjust = False).mean()

    if moving_avg == 'SMA':
        stock_df['Signal'] = np.where(stock_df[short_window_col_sma] > stock_df[long_window_col_sma], 1.0, 0.0)
    elif moving_avg == 'EMA':
        stock_df['Signal'] = np.where(stock_df[short_window_col_ema] > stock_df[long_window_col_ema], 1.0, 0.0)
    elif moving_avg == 'Both':
        stock_df['Signal'] = np.where(((stock_df[short_window_col_sma] > stock_df[long_window_col_sma]) & 
                                        (stock_df[short_window_col_ema] > stock_df[long_window_col_ema])), 1.0, 0.0)
    
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

    # Calculate Buy and Hold Strategy
    initial_qty = initial_cash // stock_df['Close Price'].iloc[0]
    final_value_hold = initial_qty * stock_df['Close Price'].iloc[-1]
    hold_return = ((final_value_hold - initial_cash) / initial_cash) * 100
    final_return = ((final_value - initial_cash) / initial_cash) * 100

    # Counting the number of trades
    df_pos = stock_df.loc[(stock_df['Position'] == 1) | (stock_df['Position'] == -1)].copy()
    num_trades = df_pos['Position'].value_counts()

    # st.write(f'Final portfolio value using strategy: final_value:  {final_value} ')
    # st.write(f'Final portfolio value using strategy: Return Percent: {final_return:.2f}%')

    # st.write(f'Final portfolio value using Buy and Hold: final_value: {final_value_hold}')
    # st.write(f'Final portfolio value using Buy and Hold: Return Percent: {hold_return:.2f}%')  

    # st.write(f"Number of Buy trades: {num_trades[1]}")
    # st.write(f"Number of Sell trades: {num_trades[-1]}")

    fig = plt.figure(figsize=(20, 10))
    plt.tick_params(axis='both', labelsize=14)
    stock_df['Close Price'].plot(color='k', lw=1, label='Close Price')  
    if moving_avg == 'SMA' or moving_avg == 'Both':
        stock_df[short_window_col_sma].plot(color='b', lw=1, label=short_window_col_sma)
        stock_df[long_window_col_sma].plot(color='g', lw=1, label=long_window_col_sma)
    if moving_avg == 'EMA' or moving_avg == 'Both':
        stock_df[short_window_col_ema].plot(color='r', lw=1, label=short_window_col_ema)
        stock_df[long_window_col_ema].plot(color='y', lw=1, label=long_window_col_ema)
    plt.plot(stock_df[stock_df['Position'] == 1].index, stock_df['Close Price'][stock_df['Position'] == 1], '^', markersize=15, color='g', alpha=0.7, label='buy')
    plt.plot(stock_df[stock_df['Position'] == -1].index, stock_df['Close Price'][stock_df['Position'] == -1], 'v', markersize=15, color='r', alpha=0.7, label='sell')
    plt.ylabel('Price in ₹', fontsize=16)
    plt.xlabel('Date', fontsize=16)
    plt.title(str(stock_symbol) + ' - ' + str(moving_avg) + ' Crossover', fontsize=20)
    plt.legend()
    plt.grid()
    st.pyplot(fig)

    # Calculate Drawdown
    stock_df['Portfolio Value'] = stock_df['Cash'] + (stock_df['Shares'] * stock_df['Close Price'])
    stock_df['Running Max'] = np.maximum.accumulate(stock_df['Portfolio Value'])
    stock_df['Drawdown'] = stock_df['Running Max'] - stock_df['Portfolio Value']
    stock_df['Drawdown Percent'] = stock_df['Drawdown'] / stock_df['Running Max'] * 100

    # Plotting Drawdown
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.fill_between(stock_df.index, stock_df['Drawdown Percent'], color='red', alpha=0.3)
    plt.title('Portfolio Drawdown')
    plt.show()

    # Print Drawdown stats
    max_dd = np.max(stock_df['Drawdown Percent'])
    print("Maximum Drawdown: %.2f%%" % max_dd)
    st.write(f"Maximum Drawdown: {max_dd:.2f}%")

    

    if display_table:
        df_pos = stock_df.loc[(stock_df['Position'] == 1) | (stock_df['Position'] == -1)].copy()
        df_pos['Position'] = df_pos['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
        df_pos.style.format({"Return": "{:.2f}%"}).background_gradient(subset=['Return'], cmap=('Reds' if x < 0 else 'Greens' for x in df_pos['Return']))
        st.dataframe(df_pos)  
    #return [moving_avg, final_value, '{:.2f}%'.format(final_return), final_value_hold, '{:.2f}%'.format(hold_return), num_trades[1], num_trades[-1]]
    return [moving_avg, final_value, '{:.2f}%'.format(final_return), final_value_hold, '{:.2f}%'.format(hold_return), num_trades[1], num_trades[-1], '{:.2f}%'.format(max_dd)]

# Streamlit app
st.title("Moving Average Crossover Strategy Simulator")
stock_symbol = st.text_input("Stock Symbol:", 'QQQ')
start_date = st.date_input("Start Date:", pd.to_datetime('2023-01-01'))
end_date = st.date_input("End Date:", pd.to_datetime('2023-08-01'))
short_window = st.slider("Short Window:", min_value=1, max_value=50, value=5, step=1)
long_window = st.slider("Long Window:", min_value=1, max_value=200, value=20, step=1)
#moving_avg = st.selectbox("Moving Average Type:", ('SMA', 'EMA', 'Both'), index=0)
display_table = st.checkbox("Display Table?", value=True)
initial_cash = st.slider("Initial Cash:", min_value=10000, max_value=100000, value=50000, step=1000)

if st.button('Run Simulation'):
    #MovingAverageCrossStrategy(stock_symbol, start_date, end_date, short_window, long_window, moving_avg, display_table, initial_cash)
    run_simulations(stock_symbol, start_date, end_date, short_window, long_window, display_table, initial_cash)

