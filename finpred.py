import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient

# Streamlit App Configuration
st.set_page_config(page_title="Stock Analysis", layout="wide")

# Streamlit App Title
st.title("Stock Analysis and Prediction App")

# User Input for Stock Ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, RELIANCE.NS):")

# User Input for Date Range
st.sidebar.header("Customize Date Range")
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp('2023-01-01'))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp('2023-12-31'))

# News API Client
newsapi = NewsApiClient(api_key='YOUR_NEWSAPI_KEY')

# Fetch Stock Data
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

# Fetch News
def fetch_news(ticker):
    query = ticker
    articles = newsapi.get_everything(q=query, language='en', sort_by='relevance', page_size=5)
    return articles['articles']

# Analyze Sentiment
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

if ticker:
    # Check for Indian Stock Ticker Suffix
    if not ticker.endswith('.NS') and 'india' in st.session_state.get('user_country', '').lower():
        st.warning("For Indian stocks, ensure the ticker ends with '.NS' (e.g., RELIANCE.NS).")

    # Fetch and display stock data
    try:
        data = fetch_stock_data(ticker, start_date, end_date)
        if data.empty:
            st.error("No data found for the specified ticker. Please verify the symbol.")
        else:
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

            # # Sentiment Analysis
            # st.write("### Sentiment Analysis")
            # try:
            #     news = fetch_news(ticker)
            #     sentiments = []
            #     for article in news:
            #         title = article['title']
            #         sentiment_score = analyze_sentiment(title)
            #         sentiments.append({'Title': title, 'Sentiment': sentiment_score})

            #     # Convert to DataFrame
            #     sentiment_df = pd.DataFrame(sentiments)
            #     st.write("Recent News and Sentiment Scores")
            #     st.dataframe(sentiment_df)

            #     # Aggregate Sentiment
            #     avg_sentiment = sentiment_df['Sentiment'].mean()
            #     st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f}")

            #     if avg_sentiment > 0:
            #         st.success("Overall Sentiment: Positive 😊")
            #     elif avg_sentiment < 0:
            #         st.error("Overall Sentiment: Negative 😔")
            #     else:
            #         st.info("Overall Sentiment: Neutral 😐")

            # except Exception as e:
            #     st.error(f"Error fetching or analyzing sentiment: {e}")

    except Exception as e:
        st.error(f"Error fetching data for {ticker.upper()}: {e}")
