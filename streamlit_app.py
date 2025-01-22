import streamlit as st
import yfinance as yf
import pandas as pd
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
        if data.empty:
            return pd.DataFrame()
        
        # Ensure we have the close price column
        if 'Close' in data.columns:
            data['Price'] = data['Close']
        elif 'Adj Close' in data.columns:
            data['Price'] = data['Adj Close']
        else:
            st.error("Unable to find price data in the downloaded dataset")
            return pd.DataFrame()
            
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_signals(df, ma_fast, ma_slow):
    """Calculate moving averages and generate entry/exit signals"""
    df = df.copy()
    
    # Calculate moving averages using pandas rolling
    df['MA_Fast'] = df['Price'].rolling(window=ma_fast).mean()
    df['MA_Slow'] = df['Price'].rolling(window=ma_slow).mean()
    
    # Initialize signals and positions
    df['Position'] = 0
    
    # Generate positions based on MA crossover
    conditions = [
        (df['MA_Fast'] > df['MA_Slow']) & (df['MA_Fast'].shift(1) <= df['MA_Slow'].shift(1)),  # Entry
        (df['MA_Fast'] < df['MA_Slow']) & (df['MA_Fast'].shift(1) >= df['MA_Slow'].shift(1))   # Exit
    ]
    choices = [1, -1]
    df['Signal'] = np.select(conditions, choices, default=0)
    
    # Calculate positions
    df['Position'] = df['Signal'].cumsum().clip(lower=0, upper=1)
    
    # Mark trade entries and exits
    df['Trade_Entry'] = df['Position'].diff() == 1
    df['Trade_Exit'] = df['Position'].diff() == -1
    
    return df

def calculate_returns(df):
    """Calculate returns and statistics for the strategy"""
    df = df.copy()
    
    # Calculate daily returns
    df['Daily_Return'] = df['Price'].pct_change()
    
    # Calculate strategy returns
    df['Strategy_Return'] = df['Daily_Return'] * df['Position'].shift(1)
    
    # Calculate cumulative returns
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    
    # Calculate drawdown
    df['Peak'] = df['Cumulative_Return'].expanding().max()
    df['Drawdown'] = (df['Cumulative_Return'] - df['Peak']) / df['Peak'] * 100
    
    return df

def analyze_trades(df, market):
    """Analyze individual trades and calculate statistics with fixed boolean operations"""
    trades = []
    df = df.copy()
    
    # Get entry and exit points
    entry_dates = df.index[df['Trade_Entry']].tolist()
    exit_dates = df.index[df['Trade_Exit']].tolist()
    
    # Add the last date as exit if we're still in a position
    if len(entry_dates) > len(exit_dates):
        exit_dates.append(df.index[-1])
    
    currency_symbol = "₹" if market == "India" else "$"
    
    # Process each trade
    for entry_date, exit_date in zip(entry_dates, exit_dates):
        entry_price = df.loc[entry_date, 'Price']
        exit_price = df.loc[exit_date, 'Price']
        
        # Get trade period data
        trade_period = df.loc[entry_date:exit_date]
        high_price = trade_period['Price'].max()
        low_price = trade_period['Price'].min()
        
        # Calculate trade metrics
        trade_return = (exit_price - entry_price) / entry_price * 100
        holding_period = (exit_date - entry_date).days / 30.44  # Convert to months
        
        trade_data = {
            'Entry_Date': entry_date,
            'Exit_Date': exit_date if exit_date != df.index[-1] else "Open",
            'Entry_Price': f"{currency_symbol}{entry_price:.2f}",
            'Exit_Price': f"{currency_symbol}{exit_price:.2f}",
            'High_Price': f"{currency_symbol}{high_price:.2f}",
            'Low_Price': f"{currency_symbol}{low_price:.2f}",
            'Return': trade_return,
            'Holding_Period': holding_period,
            'MA_Fast_Entry': df.loc[entry_date, 'MA_Fast'],
            'MA_Slow_Entry': df.loc[entry_date, 'MA_Slow']
        }
        trades.append(trade_data)
    
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
            
            # Display metrics using columns
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
                'Price': df['Price'],
                f'{ma_fast}d MA': df['MA_Fast'],
                f'{ma_slow}d MA': df['MA_Slow']
            })
            st.line_chart(chart_data)
            
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