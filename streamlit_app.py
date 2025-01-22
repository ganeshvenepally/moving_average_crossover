import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import date, datetime
import io

def convert_df_to_csv(df):
    """Convert dataframe to CSV format for download"""
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
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_signals(df, ma_fast, ma_slow):
    """Calculate moving averages and generate entry/exit signals"""
    # Calculate moving averages
    df['MA_Fast'] = ta.sma(df['Adj Close'], length=ma_fast)
    df['MA_Slow'] = ta.sma(df['Adj Close'], length=ma_slow)
    
    # Initialize signals and positions
    df['Signal'] = 0  # 0: no signal, 1: buy, -1: sell
    df['Position'] = 0  # 0: no position, 1: in position
    
    # Generate positions based on MA crossover
    position = 0
    positions = []
    
    for i in range(len(df)):
        if pd.notna(df['MA_Fast'].iloc[i]) and pd.notna(df['MA_Slow'].iloc[i]):
            if df['MA_Fast'].iloc[i] > df['MA_Slow'].iloc[i] and position == 0:
                position = 1  # Enter position
            elif df['MA_Fast'].iloc[i] < df['MA_Slow'].iloc[i] and position == 1:
                position = 0  # Exit position
                
        positions.append(position)
    
    df['Position'] = positions
    return df

def calculate_returns(df):
    """Calculate returns and statistics for the strategy"""
    df = df.copy()
    
    # Calculate daily returns
    df['Daily_Return'] = df['Adj Close'].pct_change()
    
    # Calculate strategy returns
    df['Strategy_Return'] = df['Daily_Return'] * df['Position'].shift(1)
    
    # Calculate cumulative returns
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    
    # Calculate drawdown
    df['Peak'] = df['Cumulative_Return'].expanding().max()
    df['Drawdown'] = (df['Cumulative_Return'] - df['Peak']) / df['Peak'] * 100
    
    # Mark trade entries and exits
    df['Trade_Entry'] = df['Position'].diff() == 1
    df['Trade_Exit'] = df['Position'].diff() == -1
    
    return df

def analyze_trades(df, market):
    """Analyze individual trades and calculate statistics"""
    trades = []
    entry_price = None
    entry_date = None
    
    currency_symbol = "₹" if market == "India" else "$"
    
    for date, row in df.iterrows():
        if row['Trade_Entry']:
            entry_price = row['Adj Close']
            entry_date = date
        elif row['Trade_Exit'] and entry_price is not None:
            exit_price = row['Adj Close']
            
            # Calculate trade metrics
            trade_return = (exit_price - entry_price) / entry_price * 100
            holding_period = (date - entry_date).days / 30.44  # Convert days to months
            
            # Get trade period data for additional metrics
            trade_period = df.loc[entry_date:date]
            high_price = trade_period['Adj Close'].max()
            low_price = trade_period['Adj Close'].min()
            
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
        last_price = df['Adj Close'].iloc[-1]
        trades.append({
            'Entry_Date': entry_date,
            'Exit_Date': "Open",
            'Entry_Price': f"{currency_symbol}{entry_price:.2f}",
            'Exit_Price': f"{currency_symbol}{last_price:.2f}",
            'High_Price': f"{currency_symbol}{df['Adj Close'][entry_date:].max():.2f}",
            'Low_Price': f"{currency_symbol}{df['Adj Close'][entry_date:].min():.2f}",
            'Return': (last_price - entry_price) / entry_price * 100,
            'Holding_Period': (last_date - entry_date).days / 30.44,
            'MA_Fast_Entry': df['MA_Fast'].iloc[-1],
            'MA_Slow_Entry': df['MA_Slow'].iloc[-1]
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
            
            # Display results
            st.subheader("Strategy Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            total_return = (df['Cumulative_Return'].iloc[-1] - 1) * 100
            num_trades = len(trades_df)
            win_rate = (trades_df['Return'].astype(float) > 0).mean() * 100 if not trades_df.empty else 0
            max_drawdown = df['Drawdown'].min()
            
            col1.metric("Total Return", f"{total_return:.2f}%")
            col2.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            col3.metric("Win Rate", f"{win_rate:.2f}%")
            col4.metric("Number of Trades", num_trades)
            
            # Display trades
            if not trades_df.empty:
                st.subheader("Trade Statistics")
                stat_col1, stat_col2 = st.columns(2)
                
                with stat_col1:
                    st.markdown("**Return Metrics**")
                    stats_df1 = pd.DataFrame({
                        'Metric': ['Average Return', 'Best Trade Return', 'Worst Trade Return'],
                        'Value': [
                            f"{trades_df['Return'].mean():.2f}%",
                            f"{trades_df['Return'].max():.2f}%",
                            f"{trades_df['Return'].min():.2f}%"
                        ]
                    })
                    st.dataframe(stats_df1, hide_index=True)
                
                with stat_col2:
                    st.markdown("**Time Metrics**")
                    stats_df2 = pd.DataFrame({
                        'Metric': ['Average Holding Period', 'Longest Trade', 'Shortest Trade'],
                        'Value': [
                            f"{trades_df['Holding_Period'].mean():.1f} months",
                            f"{trades_df['Holding_Period'].max():.1f} months",
                            f"{trades_df['Holding_Period'].min():.1f} months"
                        ]
                    })
                    st.dataframe(stats_df2, hide_index=True)
                
                # Display trades table
                st.subheader("Individual Trades")
                st.dataframe(trades_df.style.format({
                    'Return': '{:.2f}%',
                    'Holding_Period': '{:.1f}',
                    'MA_Fast_Entry': '{:.2f}',
                    'MA_Slow_Entry': '{:.2f}'
                }))
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    trades_csv = convert_df_to_csv(trades_df)
                    st.download_button(
                        label="Download Trades Data",
                        data=trades_csv,
                        file_name=f'{ticker}_trades.csv',
                        mime='text/csv'
                    )
                
                with col2:
                    full_data_csv = convert_df_to_csv(df)
                    st.download_button(
                        label="Download Full Analysis Data",
                        data=full_data_csv,
                        file_name=f'{ticker}_full_analysis.csv',
                        mime='text/csv'
                    )
            else:
                st.info("No trades were generated during the selected period.")

if __name__ == "__main__":
    main()