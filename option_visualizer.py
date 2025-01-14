import streamlit as st
import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

class BlackScholesModel:
    def __init__(self):
        self.S = None
        self.K = None
        self.r = None
        self.t = None
        self.sigma = None
       
    def fetch_spy_data(self):
        """Fetch current SPY data and calculate historical volatility"""
        try:
            spy = yf.Ticker("SPY")
            hist_data = spy.history(period="1d")  # Reduced to 1 day for faster loading
           
            # Get current price
            self.S = hist_data['Close'].iloc[-1]
           
            # Use a fixed volatility if can't calculate
            self.sigma = 0.20  # 20% volatility as default
           
            # Use a default risk-free rate
            self.r = 0.04  # 4% as default
           
            return {
                'current_price': self.S,
                'volatility': self.sigma,
                'risk_free_rate': self.r
            }
        except Exception as e:
            st.error(f"Error fetching market data: {e}")
            # Use default values if API fails
            self.S = 400  # Default SPY price
            self.sigma = 0.20
            self.r = 0.04
            return {
                'current_price': self.S,
                'volatility': self.sigma,
                'risk_free_rate': self.r
            }

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.t)

    def call_price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.t) * norm.cdf(d2)

    def put_price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.K * np.exp(-self.r * self.t) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def calculate_greeks(self):
        d1 = self.d1()
        d2 = self.d2()
       
        call_delta = norm.cdf(d1)
        put_delta = call_delta - 1
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.t))
        call_theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.t))
                     - self.r * self.K * np.exp(-self.r * self.t) * norm.cdf(d2))
        put_theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.t))
                    + self.r * self.K * np.exp(-self.r * self.t) * norm.cdf(-d2))
        vega = self.S * np.sqrt(self.t) * norm.pdf(d1)

        return {
            'call_delta': call_delta,
            'put_delta': put_delta,
            'gamma': gamma,
            'call_theta': call_theta / 365,
            'put_theta': put_theta / 365,
            'vega': vega / 100
        }

    def price_options(self, strike_price, days_to_expiration):
        self.K = strike_price
        self.t = days_to_expiration / 365
       
        prices = {
            'call_price': self.call_price(),
            'put_price': self.put_price()
        }
       
        greeks = self.calculate_greeks()
        return {**prices, **greeks}

def plot_options_surface(df, value_column, title):
    fig = go.Figure(data=[go.Surface(
        x=df.pivot_table(index='expiration', columns='strike', values=value_column).columns,
        y=df.pivot_table(index='expiration', columns='strike', values=value_column).index,
        z=df.pivot_table(index='expiration', columns='strike', values=value_column).values
    )])
   
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Days to Expiration',
            zaxis_title=value_column
        ),
        width=800,
        height=600
    )
   
    return fig

def main():
    st.set_page_config(page_title="SPY Options Calculator", layout="wide")
   
    # Initialize model and fetch data
    bs_model = BlackScholesModel()
    market_data = bs_model.fetch_spy_data()

    # Display current market data
    st.title("SPY Options Calculator")
    col1, col2, col3 = st.columns(3)
    col1.metric("SPY Price", f"${market_data['current_price']:.2f}")
    col2.metric("Volatility", f"{market_data['volatility']*100:.2f}%")
    col3.metric("Risk-free Rate", f"{market_data['risk_free_rate']*100:.2f}%")

    # User inputs in sidebar
    st.sidebar.header("Option Parameters")
    use_custom = st.sidebar.checkbox("Use Custom Parameters")
    if use_custom:
        bs_model.S = st.sidebar.number_input("Custom SPY Price", value=market_data['current_price'])
        bs_model.sigma = st.sidebar.number_input("Custom Volatility", value=market_data['volatility'])
        bs_model.r = st.sidebar.number_input("Custom Risk-free Rate", value=market_data['risk_free_rate'])

    # Strike and expiration inputs
    strike_range = st.sidebar.slider(
        "Strike Price Range",
        min_value=int(market_data['current_price'] - 50),
        max_value=int(market_data['current_price'] + 50),
        value=(int(market_data['current_price'] - 20), int(market_data['current_price'] + 20)),
        step=5
    )
   
    exp_range = st.sidebar.slider(
        "Days to Expiration Range",
        min_value=1, max_value=365,
        value=(7, 90), step=1
    )

    # Calculate options data
    strikes = np.arange(strike_range[0], strike_range[1] + 1, 5)
    expirations = np.arange(exp_range[0], exp_range[1] + 1, (exp_range[1] - exp_range[0]) // 4)
   
    results = []
    for expiration in expirations:
        for strike in strikes:
            option_data = bs_model.price_options(strike, expiration)
            results.append({
                'expiration': expiration,
                'strike': strike,
                **option_data
            })

    df = pd.DataFrame(results)

    # Display results
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Options Prices and Greeks")
        selected_exp = st.selectbox("Select Days to Expiration", sorted(df['expiration'].unique()))
        st.dataframe(df[df['expiration'] == selected_exp].round(3))
   
    with col2:
        st.header("Visualization")
        plot_type = st.radio(
            "Select View",
            ["Call Price", "Put Price", "Greeks"]
        )
       
        if plot_type in ["Call Price", "Put Price"]:
            column = 'call_price' if plot_type == "Call Price" else 'put_price'
            st.plotly_chart(plot_options_surface(df, column, f'{plot_type} Surface'), use_container_width=True)
        else:
            greek = st.selectbox("Select Greek", ['call_delta', 'put_delta', 'gamma', 'call_theta', 'put_theta', 'vega'])
            st.plotly_chart(plot_options_surface(df, greek, f'{greek.capitalize()} Surface'), use_container_width=True)

    # Auto-refresh
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("Auto-Refresh", value=True):
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()


