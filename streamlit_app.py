import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime
import io

def convert_df_to_csv(df):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=True)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()

# Define assets
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
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def analyze_ma_strategy(data, ma_fast, ma_slow):
    """Analyze moving average strategy for a given dataset"""
    if data is None or len(data) < ma_slow:
        return None
    
    df = data.copy()
    
    # Calculate moving averages
    df['MA_Fast'] = df['Close'].rolling(window=ma_fast).mean()
    df['MA_Slow'] = df['Close'].rolling(window=ma_slow).mean()
    
    # Calculate signals and positions
    df['Signal'] = 0
    df.loc[df['MA_Fast'] > df['MA_Slow'], 'Signal'] = 1
    df.loc[df['MA_Fast'] < df['MA_Slow'], 'Signal'] = -1
    df['Position'] = df['Signal'].shift(1)
    df['Position'] = df['Position'].fillna(0)
    
    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    
    # Calculate cumulative returns
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    
    # Calculate drawdown
    df['Peak'] = df['Cumulative_Returns'].expanding().max()
    df['Drawdown'] = (df['Cumulative_Returns'] - df['Peak']) / df['Peak'] * 100
    
    # Identify trade entry and exit points
    df['Position_Change'] = df['Position'].diff()
    df['Trade_Entry'] = df['Position_Change'] == 1
    df['Trade_Exit'] = df['Position_Change'] == -1
    
    return df

def get_performance_metrics(df):
    """Calculate performance metrics"""
    if df is None:
        return None
        
    total_return = (df['Cumulative_Returns'].iloc[-1] - 1) * 100
    max_drawdown = df['Drawdown'].min()
    
    # Calculate trade statistics
    trades = len(df[df['Position_Change'] != 0])
    winning_trades = len(df[(df['Position_Change'] != 0) & (df['Strategy_Returns'] > 0)])
    win_rate = (winning_trades / trades * 100) if trades > 0 else 0
    
    return {
        'Total Return': f"{total_return:.2f}%",
        'Max Drawdown': f"{max_drawdown:.2f}%",
        'Number of Trades': trades,
        'Win Rate': f"{win_rate:.2f}%"
    }

def create_trade_summary(df):
    """Create trade summary DataFrame"""
    trades = []
    entry_price = None
    entry_date = None
    
    for idx, row in df.iterrows():
        # Entry signal
        if row['Trade_Entry']:
            entry_date = idx
            entry_price = row['Close']
        # Exit signal
        elif row['Trade_Exit'] and entry_price is not None:
            exit_price = row['Close']
            trade_return = ((exit_price - entry_price) / entry_price) * 100
            trades.append({
                'Entry Date': entry_date.strftime('%Y-%m-%d'),
                'Exit Date': idx.strftime('%Y-%m-%d'),
                'Entry Price': f"${entry_price:.2f}",
                'Exit Price': f"${exit_price:.2f}",
                'Return (%)': f"{trade_return:.2f}%",
                'Fast MA': f"${row['MA_Fast']:.2f}",
                'Slow MA': f"${row['MA_Slow']:.2f}"
            })
            entry_price = None
    
    # Handle open position
    if entry_price is not None:
        last_row = df.iloc[-1]
        exit_price = last_row['Close']
        trade_return = ((exit_price - entry_price) / entry_price) * 100
        trades.append({
            'Entry Date': entry_date.strftime('%Y-%m-%d'),
            'Exit Date': 'Open',
            'Entry Price': f"${entry_price:.2f}",
            'Exit Price': f"${exit_price:.2f}",
            'Return (%)': f"{trade_return:.2f}%",
            'Fast MA': f"${last_row['MA_Fast']:.2f}",
            'Slow MA': f"${last_row['MA_Slow']:.2f}"
        })
    
    return pd.DataFrame(trades) if trades else pd.DataFrame()

def main():
    st.set_page_config(layout="wide")
    st.title("Moving Average Crossover Strategy Backtester")
    
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
        ma_fast = st.slider("Fast Moving Average (Days)", 5, 200, 50)
    
    with col2:
        ma_slow = st.slider("Slow Moving Average (Days)", 5, 200, 200)
    
    if st.button("Run Analysis"):
        st.subheader(f"Analysis Results for {ticker}")
        
        start_date = pd.to_datetime(end_date) - pd.DateOffset(months=lookback_months)
        
        with st.spinner('Analyzing data...'):
            # Fetch and analyze data
            data = fetch_data(ticker, start_date, end_date)
            
            if data is not None:
                results = analyze_ma_strategy(data, ma_fast, ma_slow)
                
                if results is not None:
                    # Display strategy rules
                    st.info(f"""
                    Strategy Rules:
                    - Enter when {ma_fast}-day MA crosses above {ma_slow}-day MA
                    - Exit when {ma_fast}-day MA crosses below {ma_slow}-day MA
                    """)
                    
                    # Display performance metrics
                    metrics = get_performance_metrics(results)
                    if metrics:
                        cols = st.columns(4)
                        for i, (metric, value) in enumerate(metrics.items()):
                            cols[i].metric(metric, value)
                    
                    # Plot price and moving averages
                    st.subheader("Price and Moving Averages")
                    plot_df = pd.DataFrame(index=results.index)
                    plot_df['Price'] = results['Close']
                    plot_df[f'{ma_fast}d MA'] = results['MA_Fast']
                    plot_df[f'{ma_slow}d MA'] = results['MA_Slow']
                    st.line_chart(plot_df)
                    
                    # Create and display trade summary
                    st.subheader("Trade Summary")
                    trade_summary = create_trade_summary(results)
                    
                    if not trade_summary.empty:
                        st.dataframe(trade_summary)
                        
                        # Add download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            trades_csv = convert_df_to_csv(trade_summary)
                            st.download_button(
                                label="Download Trades Data",
                                data=trades_csv,
                                file_name=f'{ticker}_trades.csv',
                                mime='text/csv'
                            )
                        
                        with col2:
                            full_data_csv = convert_df_to_csv(results)
                            st.download_button(
                                label="Download Full Analysis Data",
                                data=full_data_csv,
                                file_name=f'{ticker}_full_analysis.csv',
                                mime='text/csv'
                            )
                    else:
                        st.info("No trades were generated during the selected period.")
                else:
                    st.error("Not enough data for the selected moving average periods.")
            else:
                st.error(f"No data available for {ticker}")

if __name__ == "__main__":
    main()