import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import numpy as np

# Function to fetch stock data from Yahoo Finance API
def fetch_stock_data(stock_symbol, start_date, end_date):
    try:
        # Fetch historical stock data using yfinance
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to run Prophet model and return forecast
def run_prophet_model(stock_data):
    # Prepare DataFrame for Prophet (requires 'ds' and 'y' columns)
    df_prophet = stock_data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

    # Initialize Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)

    # Make future predictions
    future_dates = model.make_future_dataframe(periods=30)  # Forecasting for next 30 days
    forecast = model.predict(future_dates)

    return model, forecast

# Function to evaluate model performance (RMSE and MAPE)
def evaluate_model(actual, predicted):
    # Calculate Root Mean Squared Error (RMSE)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return rmse, mape

# Main function to run the Streamlit app
def main():
    # Set title and sidebar options
    st.title("Stock Data Analysis App")

    # Sidebar input fields for stock symbol and date range
    with st.sidebar:
        st.header("Stock Selection")
        stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)")
        start_date = st.date_input("Select Start Date")
        end_date = st.date_input("Select End Date")

    if stock_symbol and start_date and end_date:
        # Fetch stock data when inputs are provided
        stock_data = fetch_stock_data(stock_symbol, start_date, end_date)

        if stock_data is not None:
            st.success(f"Data successfully fetched for {stock_symbol} from {start_date} to {end_date}")

            # Sidebar menu options
            st.sidebar.subheader("Menu Options")
            menu_option = st.sidebar.radio("Select an option", ["Columns", "Overview", "Prophet Forecast", "Model Results"])

            if menu_option == "Columns":
                # Option: Display columns
                st.subheader("Columns")
                st.write(list(stock_data.columns))

                # Display DataFrame
                st.write("DataFrame:")
                st.write(stock_data)

            elif menu_option == "Overview":
                # Option: Display daily, weekly, and monthly close price trends
                st.subheader("Stock Price Overview")

                # Resample data for daily, weekly, and monthly trends
                daily_prices = stock_data['Close']
                weekly_prices = daily_prices.resample('D').mean().dropna()
                monthly_prices = daily_prices.resample('M').mean().dropna()

                # Plot daily, weekly, and monthly trends with interactive Plotly graphs
                fig = px.line(stock_data, x=stock_data.index, y='Close', title='Stock Close Price Trends')
                fig.update_xaxes(title='Date')
                fig.update_yaxes(title='Close Price')
                st.plotly_chart(fig)

                # Separate graphs for Open, Close, High
                fig_open = px.line(stock_data, x=stock_data.index, y='Open', title='Stock Open Price Trends')
                fig_open.update_xaxes(title='Date')
                fig_open.update_yaxes(title='Open Price')
                st.plotly_chart(fig_open)

                fig_high = px.line(stock_data, x=stock_data.index, y='High', title='Stock High Price Trends')
                fig_high.update_xaxes(title='Date')
                fig_high.update_yaxes(title='High Price')
                st.plotly_chart(fig_high)

            elif menu_option == "Prophet Forecast":
                # Option: Run Prophet forecasting model
                st.subheader("Prophet Forecast")

                # Run Prophet model and get forecast
                model, forecast = run_prophet_model(stock_data)

                # Display forecasted data as DataFrame
                st.write("Forecasted Data:")
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

                # Plot forecasted data
                fig = model.plot(forecast)
                plt.title("Prophet Forecast")
                plt.xlabel("Date")
                plt.ylabel("Close Price")
                st.pyplot(fig)

                # Display forecast components
                st.subheader("Forecast Components")
                fig_components = model.plot_components(forecast)
                st.pyplot(fig_components)

            elif menu_option == "Model Results":
                # Option: Display Prophet model evaluation metrics
                st.subheader("Prophet Model Evaluation Metrics")

                # Run Prophet model and get forecast
                model, forecast = run_prophet_model(stock_data)

                # Extract actual and predicted values
                actual = stock_data['Close'].values
                predicted = forecast['yhat'].values[:-30]  # Use only actual data for evaluation

                # Evaluate model performance (RMSE and MAPE)
                rmse, mape = evaluate_model(actual, predicted)

                # Display model evaluation metrics
                st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f} %")


if __name__ == "__main__":
    main()
