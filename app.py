import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import os

# Load the Pre-Trained Model
model = load_model("./Stock Predictions Model.keras")

# Streamlit UI
st.title('📈 Stock Market Predictor & Buy/Sell Recommendation')

# User Input
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start_date = st.date_input('Select Start Date', pd.to_datetime('2012-01-01'))
end_date = st.date_input('Select End Date', pd.to_datetime('2025-02-28'))

# Fetch Stock Data
data = yf.download(stock, start=start_date, end=end_date)
if data.empty:
    st.error("⚠️ Invalid stock symbol or no data available for the selected period.")
    st.stop()

st.subheader('📊 Stock Data')
st.write(data)  # Show complete data from start date to end date

# Splitting Data
data_train = pd.DataFrame(data.Close[:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

# Data Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test)

# Corrected Moving Averages with Dropped NaN and Proper Indexing
ma_50 = data['Close'].rolling(50).mean().dropna()
ma_100 = data['Close'].rolling(100).mean().dropna()
ma_200 = data['Close'].rolling(200).mean().dropna()

import matplotlib.pyplot as plt

# 📈 Corrected Plot for Price vs Moving Averages using Matplotlib
st.subheader('📈 Price vs Moving Averages (Matplotlib)')
fig, ax = plt.subplots(figsize=(12, 6))

# Plot Actual Price
ax.plot(data.index, data['Close'], label='Actual Price', color='green')

# Plot Moving Averages
ax.plot(ma_50.index, ma_50, label='50-day MA', color='red')
ax.plot(ma_100.index, ma_100, label='100-day MA', color='blue')
ax.plot(ma_200.index, ma_200, label='200-day MA', color='purple')

# Add Labels and Legend
ax.set_title('Price vs Moving Averages')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price (USD)')
ax.legend()

# Show Plot in Streamlit
st.pyplot(fig)



# Preparing Data for Prediction
x, y = [], []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i - 100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

# Make Predictions
predictions = model.predict(x)

# Reverse Scaling
scale_factor = 1 / scaler.scale_[0]
predictions = predictions * scale_factor
y = y * scale_factor

# Plot Predicted vs Actual Prices
st.subheader('📊 Predicted Price vs Actual Price')
fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=y, mode='lines', name='Actual Price'))
fig2.add_trace(go.Scatter(y=predictions.flatten(), mode='lines', name='Predicted Price', line=dict(color='red')))
st.plotly_chart(fig2)

# Prediction Decision
predicted_price = predictions[-1][0]
current_price = float(data['Close'].iloc[-1])

st.subheader("📌 Future Price Prediction")
st.write(f"🔹 **Current Price:** ${current_price:.2f}")
st.write(f"🔹 **Predicted Price:** ${predicted_price:.2f}")

if predicted_price > current_price:
    st.write("✅ **Prediction: BUY (Price Expected to Increase)**")
else:
    st.write("❌ **Prediction: SELL (Price Expected to Decrease)**")

# Moving Average Crossover Strategy
st.subheader("📊 Moving Average Buy/Sell Signal")
if not ma_50.empty and not ma_200.empty:
    last_ma_50, last_ma_200 = float(ma_50.iloc[-1]), float(ma_200.iloc[-1])
    if last_ma_50 > last_ma_200:
        st.write("✅ **BUY Signal:** (50-day MA crossed above 200-day MA)")
    elif last_ma_50 < last_ma_200:
        st.write("❌ **SELL Signal:** (50-day MA fell below 200-day MA)")
    else:
        st.write("⚖️ **HOLD:** No clear trend")
else:
    st.write("⚠️ Not enough data for Moving Average analysis.")

# Technical Indicator - RSI
rsi = RSIIndicator(data['Close'].squeeze()).rsi().dropna()

# RSI Analysis
st.subheader("📈 RSI Indicator Analysis")
if not rsi.empty:
    latest_rsi = rsi.iloc[-1]
    st.write(f"📌 RSI Value: {latest_rsi:.2f}")
    if latest_rsi > 70:
        st.write("⚠️ **Overbought Condition - SELL Signal**")
    elif latest_rsi < 30:
        st.write("✅ **Oversold Condition - BUY Signal**")
    else:
        st.write("⚖️ **Neutral - HOLD**")
else:
    st.write("⚠️ Not enough data for RSI calculation.")

# Fetch News Sentiment Analysis
st.subheader("📰 News Sentiment Analysis")
news_api_key = "51272f93a04f490cb8495d26f48d76f6" 
news_url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={news_api_key}"

try:
    response = requests.get(news_url)
    response.raise_for_status()
    news_data = response.json()
    if "articles" in news_data:
        articles = news_data["articles"][:5]
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = [analyzer.polarity_scores(article["title"])['compound'] for article in articles]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

        st.write(f"📌 **Average Sentiment Score:** {avg_sentiment:.2f}")
        if avg_sentiment > 0:
            st.write("✅ **Sentiment: BUY (Positive News)**")
        elif avg_sentiment < 0:
            st.write("❌ **Sentiment: SELL (Negative News)**")
        else:
            st.write("⚖️ **Neutral Sentiment**")

        for article in articles:
            st.write(f"- {article['title']}")
            st.markdown(f"[🔗 Open News]({article['url']})", unsafe_allow_html=True)
    else:
        st.write("⚠️ No news articles found.")

except requests.RequestException as e:
    st.error(f"⚠️ Could not fetch news data. Error: {e}")

# Final Decision
st.subheader("🤖 Final Stock Recommendation")
buy_signals, sell_signals = 0, 0

if last_ma_50 > last_ma_200:
    buy_signals += 1
elif last_ma_50 < last_ma_200:
    sell_signals += 1

if not np.isnan(latest_rsi):
    if latest_rsi < 30:
        buy_signals += 1
    elif latest_rsi > 70:
        sell_signals += 1

if predicted_price > current_price:
    buy_signals += 1
else:
    sell_signals += 1

if avg_sentiment > 0:
    buy_signals += 1
elif avg_sentiment < 0:
    sell_signals += 1

if buy_signals > sell_signals:
    st.write("✅ **Final Decision: STRONG BUY**")
elif sell_signals > buy_signals:
    st.write("❌ **Final Decision: STRONG SELL**")
else:
    st.write("⚖️ **Final Decision: HOLD**")
