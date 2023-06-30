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

    # Calculate Buy and Hold Strategy
    initial_qty = initial_cash // stock_df['Close Price'].iloc[0]
    final_value_hold = initial_qty * stock_df['Close Price'].iloc[-1]
    hold_return = ((final_value_hold - initial_cash) / initial_cash) * 100
    final_return = ((final_value - initial_cash) / initial_cash) * 100

    st.write(f'Final portfolio value using strategy: final_value:  {final_value} ')
    st.write(f'Final portfolio value using strategy: Return Percent: {final_return}%')

    st.write(f'Final portfolio value using Buy and Hold: final_value: {final_value_hold}')
    st.write(f'Final portfolio value using Buy and Hold: Return Percent: {hold_return}%')

    fig = plt.figure(figsize=(20, 10))
    plt.tick_params(axis='both', labelsize=14)
    stock_df['Close Price'].plot(color='k', lw=1, label='Close Price')  
    if moving_avg == 'SMA' or moving_avg == 'Both':
        stock_df[short_window_col_sma].plot(color='b', lw=1, label=short_window_col_sma)
        stock_df[long_window_col_sma].plot(color='g', lw=1, label=long_window_col_sma)
    if moving_avg == 'EMA' or moving_avg == 'Both':
        stock_df[short_window_col_ema].plot(color='r', lw=1, label=short_window_col_ema)
        stock_df[long_window_col_ema].plot(color='y', lw=1, label=long_window_col_ema)
    plt.plot(stock_df[stock_df['Position'] == 1].index, stock_df[short_window_col_sma][stock_df['Position'] == 1], '^', markersize=15, color='g', alpha=0.7, label='buy')
    plt.plot(stock_df[stock_df['Position'] == -1].index, stock_df[short_window_col_sma][stock_df['Position'] == -1], 'v', markersize=15, color='r', alpha=0.7, label='sell')
    plt.ylabel('Price in ₹', fontsize=16)
    plt.xlabel('Date', fontsize=16)
    plt.title(str(stock_symbol) + ' - ' + str(moving_avg) + ' Crossover', fontsize=20)
    plt.legend()
    plt.grid()
    st.pyplot(fig)

    if display_table:
        df_pos = stock_df.loc[(stock_df['Position'] == 1) | (stock_df['Position'] == -1)].copy()
        df_pos['Position'] = df_pos['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
        df_pos.style.format({"Return": "{:.2f}%"}).background_gradient(subset=['Return'], cmap=('Reds' if x < 0 else 'Greens' for x in df_pos['Return']))
        st.dataframe(df_pos)

    # Calculate portfolio returns from prices
    stock_df['Portfolio Returns'] = stock_df['Close Price'].pct_change()
    stock_df['Portfolio Returns'].fillna(0, inplace=True)
     
    return stock_df
    return stock_df, final_value

st.title("Moving Average Crossover Strategy Simulator")
stock_symbol = st.text_input("Stock Symbol:", 'ASIANPAINT.NS')
start_date = st.date_input("Start Date:", pd.to_datetime('2022-01-31'))
end_date = st.date_input("End Date:", pd.to_datetime('2023-06-01'))
short_window = st.slider('Short window period:', min_value=5, max_value=50, value=20)
long_window = st.slider('Long window period:', min_value=50, max_value=200, value=100)
moving_avg = st.selectbox('Moving Average Type:', options=['SMA', 'EMA', 'Both'], index=0)
display_table = st.checkbox("Display Buy/Sell Transactions Table", value=True)
initial_cash = st.number_input("Initial Cash:", value=100000.0, step=10000.0)

#data, final_value = MovingAverageCrossStrategy(stock_symbol, start_date, end_date, short_window, long_window, moving_avg, display_table, initial_cash)


data = MovingAverageCrossStrategy(stock_symbol, start_date, end_date, short_window, long_window, moving_avg, display_table, initial_cash)

# QuantStat report
# Get the SPY data
spy = yf.download('SPY', start=start_date, end=end_date)['Adj Close'].pct_change()

# Remove timezone information from SPY data
spy.index = spy.index.tz_localize(None)

st.header("QuantStat Report")
st.write(qs.reports.full(data['Portfolio Returns'], spy))

# QuantStat report
returns = data['Portfolio Returns']
returns.fillna(0, inplace=True)
st.header("QuantStat Report")
st.write(qs.reports.full(returns, "SPY"))
