import os
import warnings
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from sklearn.ensemble import HistGradientBoostingRegressor

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, message="Series.__getitem__ treating keys as positions is deprecated.*")

while True:
    # Ask the user for the stock ticker symbol
    stock_ticker = input("Enter the stock ticker symbol or 'exit' to finish: ")
    if stock_ticker.lower() == 'exit':
        break

    # Get today's date
    today = datetime.datetime.now().date()

    # Subtract 365 days from today's date
    one_year_ago = today - datetime.timedelta(days=365)

    # Use the date one year ago as the start parameter in yf.download()
    data = yf.download(stock_ticker, start=one_year_ago)

    if data.empty:
        print("No data available for the stock ticker symbol:", stock_ticker, ". Please try another symbol.")
        continue
    else:
        # Convert the date column to a datetime object
        data['Date'] = pd.to_datetime(data.index)

        # Set the date column as the index
        data.set_index('Date', inplace=True)

        # Sort the data by date
        data.sort_index(inplace=True)

        # Get the data for the last year
        last_year = data.iloc[-365:]

        # Split the data into X (features) and y (target)
        X = last_year.drop(columns=['Close'])
        y = last_year['Close']
        
        # Create an HistGradientBoostingRegressor instance
        model = HistGradientBoostingRegressor()

        # Fit the model with the data
        model.fit(X, y)

        # Make predictions for the next 30 days
        future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
        future_data = pd.DataFrame(index=future_dates, columns=X.columns)
        predictions = model.predict(future_data)
        
        # Calculate the standard deviation of the last year's close prices
        std_dev = last_year['Close'].std()

        # Generate random values with a standard deviation of 0.5 * the last year's close prices standard deviation
        random_values = np.random.normal(0, 0.2 * std_dev, predictions.shape)

        # Add the random values to the predicted prices
        predictions += random_values 
        predictions_df = pd.DataFrame(predictions, index=future_dates, columns=['Close'])
        
        # Concatenate the last_year and predictions dataframes
        predictions_df = pd.concat([last_year, predictions_df])

        # Set the style to dark theme
        style.use('dark_background')

        # Create the plot
        fig, ax = plt.subplots()
        ax.plot(predictions_df.index, predictions_df['Close'], color='green' if predictions_df['Close'][-1] >= last_year['Close'][-1] else 'r', label='Predicted')
        ax.plot(last_year.index, last_year['Close'], color='blue', label='Actual')
        plt.title(stock_ticker.upper() + ' Historical Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()

        # Set x-axis as date format
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%B %D %Y"))
        plt.xticks(rotation=45)

        # Show the plot
        plt.show()

