import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Function to fetch stock data
def fetch_stock_data(symbol, start, end):
    return yf.Ticker(symbol).history(start=start, end=end)

# Main function to run the Streamlit app
def main():
    st.title("Stock Data Analysis App")
    st.sidebar.image("/content/Screenshot 2024-04-14 at 2.14.01 PM.png", width=200)

    # Date range selection
    start_date = st.sidebar.date_input("Select start date", datetime(2023, 1, 1))
    end_date = st.sidebar.date_input("Select end date", datetime(2024, 1, 17))

    # Dropdown menu for each stock
    stock_symbols = ['AAPL', 'SPY', 'VXX', 'UUP', 'JNK']
    selected_stocks = []
    for i in range(5):
        selected_stock = st.sidebar.selectbox(f"Select stock {i+1}", stock_symbols)
        selected_stocks.append(selected_stock)

    # Fetch data as soon as stocks and date range are selected
    stock_data = {}
    for symbol in selected_stocks:
        data = fetch_stock_data(symbol, start_date, end_date)
        stock_data[symbol] = data

    # Create target dataframe
    target = pd.DataFrame()
    target['return'] = stock_data['AAPL']['Close'].pct_change(1) * 100  # Assuming AAPL is the target
    target = target.dropna()  # Drop NA in the first row

    # Create features dataframe
    features = pd.DataFrame()
    features['market'] = stock_data['SPY']['Close'].pct_change(1) * 100
    features['vix'] = stock_data['VXX']['Close'].diff()  # VIX is volatility index
    features['dxy'] = stock_data['UUP']['Close'].pct_change(1) * 100  # DXY is Dollar index
    features['junk'] = stock_data['JNK']['Close'].pct_change(1) * 100  # Junk bond index
    features = features.dropna()  # Drop NA in the first row

    # Radio button to display different types of data
    data_type = st.sidebar.radio("Data", options=["Stock Data", "Modeling Data", "Select Model"])

    if data_type == "Stock Data":
        # Display dataframes for selected stocks
        for symbol, data in stock_data.items():
            st.subheader(f"Stock: {symbol}")
            st.write(data)

    elif data_type == "Modeling Data":
        st.subheader("Target Data")
        st.write(target.head())

        st.subheader("Features Data")
        st.write(features.head())

    elif data_type == "Select Model":
        # Run multiple linear regression
        regression = LinearRegression()
        features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25, random_state=0)

        # Model is trained on 75% of the data
        model = regression.fit(features_train, target_train)

        st.subheader("Linear Regression Model")
        st.write("Model Intercept:", model.intercept_)
        st.write("Model Coefficients:", model.coef_)
        st.write("Training score: ", model.score(features_train, target_train))
        st.write("Test score: ", model.score(features_test, target_test))

if __name__ == "__main__":
    main()
