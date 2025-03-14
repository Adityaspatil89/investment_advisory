import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
import numpy as np
import datetime

st.set_page_config(page_title="Investment Advisor AI", layout="centered")

# -------------------------------------------
# 🔹 Title and Description
# -------------------------------------------
st.title("📈 Investment Advisor AI")
st.write("A simple AI tool for retail investors to analyze stocks 📊")

# -------------------------------------------
# 🔹 User Input
# -------------------------------------------
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", value="AAPL")
start_date = st.date_input("From", datetime.date(2022, 1, 1))
end_date = st.date_input("To", datetime.date.today())

# -------------------------------------------
# 🔹 Stock Data
# -------------------------------------------
if ticker:
    try:
        st.subheader(f"📊 Stock Price Data for {ticker.upper()}")
        data = yf.download(ticker, start=start_date, end=end_date)
        st.line_chart(data['Close'])

        # -------------------------------------------
        # 🔹 Moving Average
        # -------------------------------------------
        st.subheader("📉 Moving Average (20-day)")
        data['MA20'] = data['Close'].rolling(window=20).mean()
        st.line_chart(data[['Close', 'MA20']])

        # -------------------------------------------
        # 🔹 Simple Prediction Model
        # -------------------------------------------
        st.subheader("🤖 AI Trend Prediction")
        data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
        data = data.dropna()

        features = data[['Close']]
        target = data['Target']

        model = LogisticRegression()
        model.fit(features, target)

        next_day = model.predict([[data['Close'].iloc[-1]]])[0]

        if next_day == 1:
            st.success("📈 Prediction: Stock may go UP tomorrow")
        else:
            st.error("📉 Prediction: Stock may go DOWN tomorrow")

        # -------------------------------------------
        # 🔹 Sentiment Analysis (Fake Headlines)
        # -------------------------------------------
        st.subheader("📰 Sentiment Analysis (Demo Headlines)")

        # For demo, use static headlines (as scraping news needs API)
        headlines = [
            f"{ticker.upper()} reports record quarterly earnings!",
            f"Analysts downgrade {ticker.upper()} amid market fears",
            f"{ticker.upper()} announces breakthrough product"
        ]

        sentiment_scores = []
        for headline in headlines:
            blob = TextBlob(headline)
            score = blob.sentiment.polarity
            sentiment_scores.append(score)
            st.write(f"📝 {headline}")
            st.write(f"➡️ Sentiment Score: {score:.2f}")

        avg_sentiment = np.mean(sentiment_scores)
        if avg_sentiment > 0.1:
            st.success("🧠 Sentiment: Positive")
        elif avg_sentiment < -0.1:
            st.error("🧠 Sentiment: Negative")
        else:
            st.warning("🧠 Sentiment: Neutral")

        # -------------------------------------------
        # 🔹 Investment Suggestion
        # -------------------------------------------
        st.subheader("📢 Investment Suggestion")
        if avg_sentiment > 0.1 and next_day == 1:
            st.success("✅ Suggestion: Consider Buying")
        elif avg_sentiment < -0.1 and next_day == 0:
            st.error("❌ Suggestion: Consider Selling")
        else:
            st.info("🤔 Suggestion: Hold / Wait")

    except Exception as e:
        st.error("Error loading stock data. Please check ticker.")
        st.code(e)

