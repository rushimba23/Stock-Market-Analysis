import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import datetime

st.title("Nifty Fifty Stock Analysis with LSTM Prediction")

# Define the Nifty-Fifty sectors and stock tickers
sectors = {
    'IT': ['INFY.NS', 'TCS.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS'],
    'Automobile': ['TATAMOTORS.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'ASHOKLEY.NS', 'MARUTI.NS', 'TVSMOTOR.NS', 'M&M.NS'],
    'Energy': ['RELIANCE.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'BPCL.NS', 'COALINDIA.NS'],
    'Finance': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'HDFC.NS', 'LT.NS', 'HDFCLIFE.NS'],
    'Consumer Goods': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'ASIANPAINT.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS'],
    'Others': ['GRASIM.NS', 'ULTRACEMCO.NS', 'TATASTEEL.NS', 'HINDALCO.NS', 'ADANIPORTS.NS', 'ADANIENT.NS', 'SHREECEM.NS', 'TITAN.NS', 'UPL.NS']
}

# Create a list of stocks with sector and name
stock_options = []
for sector, tickers in sectors.items():
    for ticker in tickers:
        stock_options.append(f"{sector} - {ticker}")

# Sidebar for user inputs
st.sidebar.header("Select Companies")
selected_stocks = st.sidebar.multiselect("Select companies", stock_options)
start_date = st.sidebar.date_input("Start Date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Function to fetch stock data
@st.cache_data
def get_stock_data(tickers, start, end):
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker.split(" - ")[1], start=start, end=end)
            df['Returns'] = df['Adj Close'].pct_change().fillna(0)
            data[ticker] = df
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            return None
    return data

if selected_stocks:
    stock_data = get_stock_data(selected_stocks, start_date, end_date)

    if stock_data:
        # Display data and plots for each company
        for ticker, df in stock_data.items():
            st.subheader(f"Data and Analysis for {ticker}")

            # Display DataFrame
            st.write(f"Historical Data for {ticker}")
            st.dataframe(df)

            # Plot Historical Prices (Non-Interactive)
            st.write(f"Historical Closing Prices for {ticker}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df.index, df['Close'], label=f"{ticker} Closing Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Closing Price")
            ax.legend()
            st.pyplot(fig)

            # Plot Returns (Interactive)
            st.write(f"Returns for {ticker}")
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Scatter(x=df.index, y=df['Returns'], mode='lines', name=f"{ticker} Returns"))
            fig_returns.update_layout(title=f"Returns for {ticker}", xaxis_title='Date', yaxis_title='Returns')
            st.plotly_chart(fig_returns)

        # Plot Comparison of All Stocks
        st.subheader("Comparison of Nifty Fifty Stocks")
        fig_compare, ax_compare = plt.subplots(figsize=(12, 8))
        for ticker, df in stock_data.items():
            if not df.empty and 'Close' in df.columns and not df['Close'].empty:
                normalized_data = (df['Close'] / df['Close'].iloc[0]) * 100
                ax_compare.plot(df.index, normalized_data, label=ticker)
            else:
                st.error(f"Data issues for {ticker}. Skipping normalization.")

        ax_compare.set_xlabel("Date")
        ax_compare.set_ylabel("Normalized Closing Price (%)")
        ax_compare.legend()
        st.pyplot(fig_compare)

        # ...existing code...
# LSTM Prediction for a Selected Stock
st.subheader("LSTM Stock Price Prediction")
selected_ticker = st.selectbox("Select Ticker for Prediction", selected_stocks)

if selected_ticker:
    df = stock_data[selected_ticker]
    if not df.empty and 'Adj Close' in df.columns:
        adj_close_prices = df['Adj Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(adj_close_prices)

        # Split data into training and testing sets
        training_data_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:training_data_len]
        test_data = scaled_data[training_data_len - 60:]

        # Create training datasets
        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

        # Build the LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        with st.spinner("Training the LSTM model..."):
            model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=0)

        # Create testing datasets
        x_test, y_test = [], adj_close_prices[training_data_len:]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test).reshape((len(x_test), 60, 1))

        # Make predictions
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Calculate MSE
        mse = mean_squared_error(y_test, predictions)

        # Generate future dates for the next 24 months
        last_date = df.index[-1]
        future_dates = pd.date_range(last_date, periods=24, freq='M')

        # Prepare data for future predictions
        future_predictions = []
        last_60_days = scaled_data[-60:]

        for i in range(24):
            future_x_test = last_60_days.reshape((1, 60, 1))
            future_pred = model.predict(future_x_test)
            future_pred_rescaled = scaler.inverse_transform(future_pred)
            future_predictions.append(future_pred_rescaled[0, 0])

            # Update last_60_days for the next prediction
            last_60_days = np.append(last_60_days[1:], future_pred, axis=0)

        # Plot actual vs predicted prices
        st.write(f"LSTM Prediction for {selected_ticker}")
        fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
        ax_pred.plot(y_test, label="Actual Prices", color="blue")
        ax_pred.plot(predictions, label="Predicted Prices", color="orange")
        ax_pred.set_xlabel("Time")
        ax_pred.set_ylabel("Stock Price")
        ax_pred.legend()
        st.pyplot(fig_pred)

        # Display MSE
        st.write(f"Mean Squared Error (MSE) for LSTM: {mse}")

        # Display future predictions in a table
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Adj Close': future_predictions
        })
        st.write(f"LSTM Prediction for Next 24 Months for {selected_ticker}")
        st.dataframe(future_df)

        # Advanced Analytics for the same selected ticker
        st.subheader("Advanced Analytics")

        if not df.empty and 'Adj Close' in df.columns:
            adj_close_prices = df['Adj Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(adj_close_prices)

            # Split data into training and testing sets
            training_data_len = int(len(scaled_data) * 0.8)
            train_data = scaled_data[:training_data_len]
            test_data = scaled_data[training_data_len - 60:]

            # Create training datasets
            x_train, y_train = [], []
            for i in range(60, len(train_data)):
                x_train.append(train_data[i-60:i, 0])
                y_train.append(train_data[i, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)

            # Create testing datasets
            x_test, y_test = [], adj_close_prices[training_data_len:]
            for i in range(60, len(test_data)):
                x_test.append(test_data[i-60:i, 0])

            x_test = np.array(x_test)

            # Hyperparameter tuning for Random Forest
            rf_param_grid = {
                'n_estimators': [100, 200],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            rf_grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=rf_param_grid, cv=3, n_jobs=-1, verbose=2)
            rf_grid_search.fit(x_train, y_train)
            rf_model = rf_grid_search.best_estimator_

            rf_predictions = rf_model.predict(x_test)
            rf_predictions = scaler.inverse_transform(rf_predictions.reshape(-1, 1))

            # Calculate MSE for Random Forest
            rf_mse = mean_squared_error(y_test, rf_predictions)

            # Hyperparameter tuning for XGBoost
            xgb_param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6, 10],
                'subsample': [0.8, 1.0]
            }
            xgb_grid_search = GridSearchCV(estimator=XGBRegressor(), param_grid=xgb_param_grid, cv=3, n_jobs=-1, verbose=2)
            xgb_grid_search.fit(x_train, y_train)
            xgb_model = xgb_grid_search.best_estimator_

            xgb_predictions = xgb_model.predict(x_test)
            xgb_predictions = scaler.inverse_transform(xgb_predictions.reshape(-1, 1))

            # Calculate MSE for XGBoost
            xgb_mse = mean_squared_error(y_test, xgb_predictions)

            # Generate future dates for the next 24 months
            future_dates = pd.date_range(df.index[-1], periods=24, freq='M')

            # Prepare data for future predictions for Random Forest
            future_rf_predictions = []
            last_60_days_rf = scaled_data[-60:]

            for i in range(24):
                future_x_test_rf = last_60_days_rf.reshape((1, 60))
                future_pred_rf = rf_model.predict(future_x_test_rf)
                future_pred_rf_rescaled = scaler.inverse_transform(future_pred_rf.reshape(-1, 1))
                future_rf_predictions.append(future_pred_rf_rescaled[0, 0])

                # Update last_60_days_rf for the next prediction
                last_60_days_rf = np.append(last_60_days_rf[1:], future_pred_rf.reshape(1, -1), axis=0)

            # Prepare data for future predictions for XGBoost
            future_xgb_predictions = []
            last_60_days_xgb = scaled_data[-60:]

            for i in range(24):
                future_x_test_xgb = last_60_days_xgb.reshape((1, 60))
                future_pred_xgb = xgb_model.predict(future_x_test_xgb)
                future_pred_xgb_rescaled = scaler.inverse_transform(future_pred_xgb.reshape(-1, 1))
                future_xgb_predictions.append(future_pred_xgb_rescaled[0, 0])

                # Update last_60_days_xgb for the next prediction
                last_60_days_xgb = np.append(last_60_days_xgb[1:], future_pred_xgb.reshape(1, -1), axis=0)

            # Plot Random Forest Predictions
            st.write(f"Random Forest Prediction for {selected_ticker}")
            fig_rf, ax_rf = plt.subplots(figsize=(10, 6))
            ax_rf.plot(y_test, label="Actual Prices", color="blue")
            ax_rf.plot(rf_predictions, label="Predicted Prices", color="green")
            ax_rf.set_xlabel("Time")
            ax_rf.set_ylabel("Stock Price")
            ax_rf.legend()
            st.pyplot(fig_rf)

            # Display MSE for Random Forest
            st.write(f"Mean Squared Error (MSE) for Random Forest: {rf_mse}")

            # Display future predictions for Random Forest in a table
            future_rf_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Adj Close': future_rf_predictions
            })
            st.write(f"Random Forest Prediction for Next 24 Months for {selected_ticker}")
            st.dataframe(future_rf_df)

            # Plot XGBoost Predictions
            st.write(f"XGBoost Prediction for {selected_ticker}")
            fig_xgb, ax_xgb = plt.subplots(figsize=(10, 6))
            ax_xgb.plot(y_test, label="Actual Prices", color="blue")
            ax_xgb.plot(xgb_predictions, label="Predicted Prices", color="red")
            ax_xgb.set_xlabel("Time")
            ax_xgb.set_ylabel("Stock Price")
            ax_xgb.legend()
            st.pyplot(fig_xgb)

            # Display MSE for XGBoost
            st.write(f"Mean Squared Error (MSE) for XGBoost: {xgb_mse}")

            # Display future predictions for XGBoost in a table
            future_xgb_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Adj Close': future_xgb_predictions
            })
            st.write(f"XGBoost Prediction for Next 24 Months for {selected_ticker}")
            st.dataframe(future_xgb_df)

            # Combine predictions into a single DataFrame
            combined_df = pd.DataFrame({
                'Date': future_dates,
                'LSTM Prediction': future_predictions,
                'Random Forest Prediction': future_rf_predictions,
                'XGBoost Prediction': future_xgb_predictions
            })

            # Display combined predictions
            st.write("Combined Predictions for Next 24 Months")
            st.dataframe(combined_df)

            # Provide option to download combined predictions as CSV
            csv = combined_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Combined Predictions as CSV",
                data=csv,
                file_name='combined_predictions.csv',
                mime='text/csv'
            )

        else:
            st.error(f"No valid data for {selected_ticker}.")
else:
    st.error("No data available. Please check the inputs.")
