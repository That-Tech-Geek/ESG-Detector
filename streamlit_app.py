import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time

# Define a default list of ticker symbols
tickers = [
    "INFY.NS", "M&M.NS", "TECHM.NS", "ADANIPORTS.NS", "MARICO.NS",
    "TATACONSUM.NS", "TCS.NS", "DRREDDY.NS", "KANSAINER.NS", "ULTRACEMCO.NS",
    "LT.NS", "AMBUJACEM.NS", "ITC.NS", "HINDZINC.NS", "ASIANPAINT.NS", "WIPRO.NS",
    "ICICIGI.NS", "HAVELLS.NS", "NAUKRI.NS", "CIPLA.NS", "GODREJCP.NS", "TATASTEEL.NS",
    "HINDALCO.NS", "MARUTI.NS", "TITAN.NS", "HCLTECH.NS", "PAGEIND.NS",
    "PIIND.NS", "ACC.NS", "HEROMOTOCO.NS", "DABUR.NS", "TATAMOTORS.NS", "SHREECEM.NS",
    "EICHERMOT.NS", "NESTLEIND.NS", "BIOCON.NS", "DIVISLAB.NS", "JSWSTEEL.NS", "PEL.NS",
    "HDFCBANK.NS", "INDUSINDBK.NS", "ICICIPRULI.NS", "WHIRLPOOL.NS", "AXISBANK.NS",
    "HINDUNILVR.NS", "GRASIM.NS", "VEDL.NS"
]

def download_and_clean_data(tickers, period="1y", interval="1d", retries=3):
    data_dict = {}
    failed_tickers = []

    for ticker in tickers:
        for attempt in range(retries):
            try:
                data = yf.download(ticker, period=period, interval=interval)['Adj Close']
                if data.isna().sum().sum() == 0:  # Ensure no NaNs in data
                    data_dict[ticker] = data
                    break  # Exit retry loop on success
                else:
                    failed_tickers.append(ticker)
                    break  # Exit retry loop on NaN data
            except Exception as e:
                if attempt < retries - 1:  # If not the last attempt, wait and retry
                    time.sleep(1)  # Wait before retrying
                else:
                    failed_tickers.append(ticker)
                    print(f"Error for {ticker} after {retries} attempts: {e}")

    if not data_dict:
        return None, failed_tickers  # Return failed tickers for user feedback

    # Create DataFrame from valid data
    try:
        data = pd.DataFrame(data_dict)
        data = data.dropna()  # Drop rows with any NaN values if present
        return data, list(data_dict.keys())
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        return None, []

# Function to calculate portfolio performance
def portfolio_performance(weights, returns, cov_matrix):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return portfolio_return, portfolio_stddev

# Sharpe ratio optimization
def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.03):
    p_return, p_stddev = portfolio_performance(weights, returns, cov_matrix)
    sharpe_ratio = (p_return - risk_free_rate) / p_stddev
    return -sharpe_ratio

# Maximize Sharpe Ratio
def maximize_sharpe_ratio(data):
    returns = data.pct_change(fill_method=None).dropna()
    cov_matrix = returns.cov()
    num_assets = len(data.columns)
    initial_guess = num_assets * [1. / num_assets]
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    result = minimize(negative_sharpe_ratio, initial_guess, args=(returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        return None, None, None, None

    optimal_weights = result.x
    optimal_return, optimal_stddev = portfolio_performance(optimal_weights, returns, cov_matrix)
    optimal_sharpe = -result.fun
    return optimal_weights, optimal_sharpe, optimal_return, optimal_stddev

# Streamlit App
st.title("Portfolio Optimization App")
st.sidebar.header("Configuration")

# Ticker Selection
selected_tickers = st.sidebar.multiselect("Select Tickers", options=tickers, default=tickers[:10])
if not selected_tickers:
    st.error("Please select at least one ticker.")

# Download and Clean Data
if st.sidebar.button("Optimize Portfolio"):
    with st.spinner("Downloading and processing data..."):
        data, failed_tickers = download_and_clean_data(selected_tickers)
        if data is None or data.empty:
            st.error("No valid data available for the selected tickers.")
            if failed_tickers:
                st.warning(f"Failed to download data for the following tickers: {', '.join(failed_tickers)}")
        else:
            # Check if there are at least 2 valid tickers
            valid_tickers = list(data.columns)
            if len(valid_tickers) < 2:
                st.error("At least 2 valid tickers are required for portfolio optimization.")
            else:
                optimal_weights, optimal_sharpe, optimal_return, optimal_stddev = maximize_sharpe_ratio(data)
                if optimal_weights is None:
                    st.error("Portfolio optimization failed.")
                else:
                    # Display results
                    st.success("Portfolio optimization complete!")
                    st.subheader("Optimization Results")
                    st.write(f"**Maximized Sharpe Ratio:** {optimal_sharpe:.4f}")
                    st.write(f"**Annualized Return:** {optimal_return:.4%}")
                    st.write(f"**Annualized Volatility:** {optimal_stddev:.4%}")

                    # Show optimal weights in a table
                    weights_df = pd.DataFrame({
                        "Ticker": valid_tickers,
                        "Weight": [f"{w:.4%}" for w in optimal_weights]
                    })
                    st.write("**Optimal Weights:**")
                    st.dataframe(weights_df)

else:
    st.info("Configure settings and click 'Optimize Portfolio' to begin.")
