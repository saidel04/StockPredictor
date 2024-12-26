import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header
        for row in csvreader:
            date = datetime.strptime(row[0], "%m/%d/%Y")
            dates.append(date)
            prices.append(float(row[1].replace('$', '')))
    return

def create_features(dates):
    return np.array([[date.toordinal(), date.month, date.day, date.weekday()] for date in dates])

def train_model(dates, prices):
    dates_features = create_features(dates)
    scaler = StandardScaler()
    dates_scaled = scaler.fit_transform(dates_features)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2)
    model.fit(dates_scaled, prices)
    return model, scaler

def predict_prices_for_week(model, scaler, start_date):
    predictions = []
    current_date = start_date
    for i in range(7):  # Predict for the next 7 days
        current_date += timedelta(days=1)
        predicted_price = predict_price(model, scaler, current_date)
        dates.append(current_date)
        prices.append(predicted_price)
        predictions.append((current_date, predicted_price))
    return predictions

def predict_price(model, scaler, date):
    date_features = create_features([date])[0]
    date_scaled = scaler.transform([date_features])
    return model.predict(date_scaled)[0]

# Get the data
get_data('aapl.csv')

# Reverse the lists to have the correct order
dates.reverse()
prices.reverse()

# Train the model
model, scaler = train_model(dates, prices)

# Predict prices for the next week
today = max(dates)
predictions = predict_prices_for_week(model, scaler, today)

# Plotting the most recent 30 days
plot_start_date = today - timedelta(days=29)
recent_dates = [date for date in dates if date >= plot_start_date]
recent_prices = prices[-len(recent_dates):]

plt.figure(figsize=(10, 6))
plt.scatter(recent_dates, recent_prices, color='black', label='Data')  # Historical data
plt.plot(recent_dates, recent_prices, color='gray')  # Line for historical data
# Add predictions to the plot
for date, price in predictions:
    plt.scatter(date, price, color='red', label='Predicted Price' if date == predictions[0][0] else "")
plt.plot(recent_dates + [date for date, _ in predictions], recent_prices + [price for _, price in predictions], color='blue', linestyle='dashed', label='Predicted Trend')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()

# Format the dates on the x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()

# Print predictions
for date, price in predictions:
    print(f'Predicted price for {date.strftime("%m/%d/%Y")}: {price}')
