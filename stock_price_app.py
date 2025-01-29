import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Application title
st.title("Stock Price Prediction Web App")

# Sidebar
st.sidebar.header("User Input Parameters")

# User input for stock ticker
stock_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

# Fetch stock data
@st.cache
def fetch_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

if st.sidebar.button("Fetch Data"):
    data = fetch_stock_data(stock_ticker, start_date, end_date)

    if data is not None and not data.empty:
        st.subheader(f"Stock Price Data for {stock_ticker}")
        st.write(data.tail())

        # Plot stock prices
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close Price"))
        fig.update_layout(title=f"{stock_ticker} Closing Prices", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

        # Add features for ML
        data["Date"] = data.index
        data["Day"] = data["Date"].apply(lambda x: x.toordinal())

        # Train-Test Split
        X = data[["Day"]]
        y = data["Close"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear Regression Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        data["Predicted"] = model.predict(X)

        # Plot actual vs predicted prices
        st.subheader("Actual vs Predicted Prices")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Actual Price"))
        fig.add_trace(go.Scatter(x=data["Date"], y=data["Predicted"], mode="lines", name="Predicted Price"))
        fig.update_layout(title=f"{stock_ticker} Price Prediction", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

        # Display model performance
        mse = mean_squared_error(y_test, model.predict(X_test))
        st.write(f"Model Performance: Mean Squared Error (MSE): {mse:.2f}")

    else:
        st.error("No data available. Please check the stock ticker or date range.")