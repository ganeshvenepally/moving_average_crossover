import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, datetime
import io
import numpy as np

def convert_df_to_csv(df):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=True)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()

# Define the assets with both US and Indian stocks
US_ASSETS = [
    "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NFLX", "NVDA", "TSLA", "COST", "JPM", "V", "JNJ"
]

INDIAN_ASSETS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS"
]

def fetch_data(ticker, start_date, end_date):
    """Fetch data for a single ticker"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_signals(df, ma_fast, ma_slow):
    """Calculate moving averages and generate entry/exit signals"""
    df = df.copy()
    
    # Calculate moving averages
    df['MA_Fast'] = df['Close'].rolling(window=ma_fast).mean()
    df['MA_Slow'] = df['Close'].rolling(window=ma_slow).mean()
    
    # Initialize signals and positions
    df['Signal'] = 0  # 0: no signal, 1: buy, -1: sell
    df['Position'] = 0  # 0: no position, 1: in position
    
    # Generate positions
    position = 0
    positions = []
    signals = []
    
    for i in range(len(df)):
        if pd.notna(df['MA_Fast'].iloc[i]) and pd.notna(df['MA_Slow'].iloc[i]):
            # Entry condition
            if df['MA_Fast'].iloc[i] > df['MA_Slow'].iloc[i] and position == 0:
                signals.append(1)
                position = 1
            # Exit condition
            elif df['MA_Fast'].iloc[i] < df['MA_Slow'].iloc[i] and position == 1:
                signals.append(-1)
                position = 0
            else:
                signals.append(0)
        else:
            signals.append(0)
        
        positions.append(position)
    
    df['Signal'] = signals
    df['Position'] = positions
    
    # Mark trade entries and exits
    df['Trade_Entry'] = (df['Position'].diff() == 1)
    df['Trade_Exit'] = (df['Position'].diff() == -1)
    
    return df

def analyze_trades(df, market):
    """Analyze individual trades and calculate statistics"""
    trades = []
    entry_price = None
    entry_date = None
    
    currency_symbol = "₹" if market == "India" else "$"
    
    for date, row in df.iterrows():
        if row['Trade_Entry'] == True:  # Explicit boolean comparison
            entry_price = row['Close']
            entry_date = date
        elif row['Trade_Exit'] == True and entry_price is not None:  # Explicit boolean comparison
            exit_price = row['Close']
            
            # Calculate trade metrics
            trade_return = (exit_price - entry_price) / entry_price * 100
            holding_period = (date - entry_date).days / 30.44  # Convert days to months
            
            # Get trade period data for additional metrics
            trade_period = df.loc[entry_date:date]
            high_price = trade_period['Close'].max()
            low_price = trade_period['Close'].min()
            
            trades.append({
                'Entry_Date': entry_date,
                'Exit_Date': date,
                'Entry_Price': f"{currency_symbol}{entry_price:.2f}",
                'Exit_Price': f"{currency_symbol}{exit_price:.2f}",
                'High_Price': f"{currency_symbol}{high_price:.2f}",
                'Low_Price': f"{currency_symbol}{low_price:.2f}",
                'Return': trade_return,
                'Holding_Period': holding_period,
                'MA_Fast_Entry': row['MA_Fast'],
                'MA_Slow_Entry': row['MA_Slow']
            })
            entry_price = None
    
    # Add last open position if exists
    if entry_price is not None:
        last_date = df.index[-1]
        last_price = df['Close'].iloc[-1]
        trades.append({
            'Entry_Date': entry_date,
            'Exit_Date': "Open",
            'Entry_Price': f"{currency_symbol}{entry_price:.2f}",
            'Exit_Price': f"{currency_symbol}{last_price:.2f}",
            'High_Price': f"{currency_symbol}{df['Close'][entry_date:].max():.2f}",
            'Low_Price': f"{currency_symbol}{df['Close'][entry_date:].min():.2f}",
            'Return': (last_price - entry_price) / entry_price * 100,
            'Holding_Period': (last_date - entry_date).days / 30.44,
            'MA_Fast_Entry': df.loc[entry_date, 'MA_Fast'],
            'MA_Slow_Entry': df.loc[entry_date, 'MA_Slow']
        })
    
    return pd.DataFrame(trades)

def main():
    st.set_page_config(layout="wide")
    st.title("Moving Average Crossover Strategy Backtester")
    
    # Input parameters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        market = st.selectbox("Select Market", ["US", "India"])
        assets = US_ASSETS if market == "US" else INDIAN_ASSETS
    
    with col2:
        ticker = st.selectbox("Select Asset", assets)
    
    with col3:
        end_date = st.date_input("End Date", date.today())
    
    with col4:
        lookback_months = st.slider("Lookback Period (Months)", 1, 60, 12)
    
    # Moving average parameters
    col1, col2 = st.columns(2)
    
    with col1:
        ma_fast = st.slider("Fast Moving Average Period (Days)", 5, 200, 50)
    
    with col2:
        ma_slow = st.slider("Slow Moving Average Period (Days)", 5, 200, 200)
        
    if st.button("Run Analysis"):
        st.subheader(f"Analysis Results for {ticker}")
        
        start_date = pd.to_datetime(end_date) - pd.DateOffset(months=lookback_months)
        
        with st.spinner('Analyzing data...'):
            # Fetch stock data
            df = fetch_data(ticker, start_date, end_date)
            
            if df.empty:
                st.error(f"No data available for {ticker}")
                return
            
            # Calculate signals and returns
            df = calculate_signals(df, ma_fast, ma_slow)
            df = calculate_returns(df)
            trades_df = analyze_trades(df, market)
            
            # Display strategy explanation
            st.info(f"""
            Strategy Rules:
            - Enter when {ma_fast}-day MA crosses above {ma_slow}-day MA
            - Exit when {ma_fast}-day MA crosses below {ma_slow}-day MA
            """)
            
            # Display metrics
            st.subheader("Strategy Performance")
            metric_cols = st.columns(4)
            
            total_return = (df['Cumulative_Return'].iloc[-1] - 1) * 100
            num_trades = len(trades_df)
            win_rate = (trades_df['Return'].astype(float) > 0).mean() * 100 if not trades_df.empty else 0
            max_drawdown = df['Drawdown'].min()
            
            metric_cols[0].metric("Total Return", f"{total_return:.2f}%")
            metric_cols[1].metric("Max Drawdown", f"{max_drawdown:.2f}%")
            metric_cols[2].metric("Win Rate", f"{win_rate:.2f}%")
            metric_cols[3].metric("Number of Trades", num_trades)
            
            # Plot price and moving averages
            st.subheader("Price and Moving Averages")
            chart_data = pd.DataFrame({
                'Price': df['Close'],
                f'{ma_fast}d MA': df['MA_Fast'],
                f'{ma_slow}d MA': df['MA_Slow']
            })
            st.line_chart(chart_data)
            
            # Display trade statistics and tables
            if not trades_df.empty:
                # [Rest of the display code remains the same]
                pass
            else:
                st.info("No trades were generated during the selected period.")

if __name__ == "__main__":
    main()