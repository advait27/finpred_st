import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Streamlit App Configuration
st.set_page_config(page_title="Stock Analysis", layout="wide")

# Streamlit App Title
st.title("Stock Analysis and Market Predictions")

# User Input for Stock Ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):")

# User Input for Date Range
st.sidebar.header("Customize Date Range")
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp('2023-01-01'))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp('2023-12-31'))

# Fetch Stock Data
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

if ticker:
    # Fetch and display stock data
    try:
        data = fetch_stock_data(ticker, start_date, end_date)
        st.write(f"### Stock Data for {ticker.upper()}")
        st.dataframe(data.tail())

        # Current Position
        st.write("### Current Stock Position")
        st.write(f"**Last Close Price:** {data['Close'][-1]:.2f}")
        st.write(f"**Open Price Today:** {data['Open'][-1]:.2f}")
        st.write(f"**High Today:** {data['High'][-1]:.2f}")
        st.write(f"**Low Today:** {data['Low'][-1]:.2f}")

        # Candlestick Chart
        st.write("### Candlestick Chart")
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])
        st.plotly_chart(fig)

        # Add Indicators
        st.write("### Indicator Graphs")
        data['MA50'] = data['Close'].rolling(50).mean()
        data['MA200'] = data['Close'].rolling(200).mean()

        fig, ax = plt.subplots()
        ax.plot(data.index, data['Close'], label='Close Price')
        ax.plot(data.index, data['MA50'], label='50-Day MA')
        ax.plot(data.index, data['MA200'], label='200-Day MA')
        ax.legend()
        st.pyplot(fig)

        # Performance Metrics
        st.write("### Performance Metrics")
        data['Daily Return'] = data['Close'].pct_change()
        st.write(f"**Annualized Volatility:** {np.std(data['Daily Return']) * np.sqrt(252):.2f}")
        st.write(f"**Yearly Return:** {((data['Close'][-1] / data['Close'][0]) - 1) * 100:.2f}%")

        # Prediction Section
        st.write("### Predict Future Prices")
        prediction_horizon = st.radio(
            "Select Prediction Horizon:",
            ('1 Week', '1 Month', '3 Months')
        )

        # Simple Linear Regression for Prediction
        if prediction_horizon:
            st.write(f"Predicting prices for: {prediction_horizon}")
            horizon_map = {'1 Week': 5, '1 Month': 20, '3 Months': 60}
            horizon = horizon_map[prediction_horizon]

            # Prepare Data
            data['Day'] = np.arange(len(data))
            X = data[['Day']]
            y = data['Close']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict Future Prices
            future_days = np.arange(len(data), len(data) + horizon).reshape(-1, 1)
            future_prices = model.predict(future_days)

            # Display Predictions
            st.write("### Predicted Prices")
            future_df = pd.DataFrame({
                'Day': future_days.flatten(),
                'Predicted Price': future_prices
            })
            st.dataframe(future_df)

            # Plot Predictions
            fig, ax = plt.subplots()
            ax.plot(data['Day'], data['Close'], label='Historical Prices')
            ax.plot(future_df['Day'], future_df['Predicted Price'], label='Predicted Prices', linestyle='--')
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error fetching data for {ticker.upper()}: {e}")
