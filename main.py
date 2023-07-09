import pandas as pd
from meteostat import Stations, Hourly, Point
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# CSV-Datei laden
bike_data = pd.read_csv('bikes.csv').dropna()
bike_data['Start Time'] = pd.to_datetime(bike_data['Start Time'])
bike_data['day'] = bike_data['Start Time'].dt.day
bike_data['hour'] = bike_data['Start Time'].dt.hour
bike_data["month"] = bike_data["Start Time"].dt.month
bike_data["weekday"] = bike_data["Start Time"].dt.day_of_week

bike_data = bike_data.set_index('Start Time')
# Fetch weather data for Toronto using Meteostat
location = Point(43.70, -79.42)  # Toronto coordinates
start = datetime(bike_data.index.min().year, bike_data.index.min().month, bike_data.index.min().day)
end = datetime(bike_data.index.max().year, bike_data.index.max().month, bike_data.index.max().day)
weather_data = Hourly(location, start, end)
weather_data = weather_data.fetch()
print(weather_data.index)

weather_data.index = pd.to_datetime(weather_data.index)
weather_data["month"] = weather_data.index.month
weather_data['day'] = weather_data.index.day
weather_data['hour'] = weather_data.index.hour
weather_data["weekday"] = weather_data.index.day_of_week

weather_data = weather_data.drop(["rhum", "prcp", "snow", "wpgt", "pres", "tsun", "coco"], axis=1)

print(weather_data)
bike_count_per_hour = bike_data.groupby(["month", "day", "hour", "weekday"]).count()['Trip Id']
print(bike_count_per_hour)
bike_count_per_hour = bike_count_per_hour.rename('Bike Count')
# Convert bike_count_per_hour Series to DataFrame
bike_count_per_hour = bike_count_per_hour.to_frame(name='Bike Count')
bike_count_per_hour.reset_index(inplace=True)

print(weather_data)
# Merge weather data with bike count per hour
merged_df = pd.merge(weather_data, bike_count_per_hour, on=["month", 'day', 'hour', "weekday"], how='inner')

columns_to_scale = ['temp', 'dwpt', 'wdir', 'wspd']
subset = merged_df[columns_to_scale]

# Rescale weather data
scaler = MinMaxScaler()
weather_features = scaler.fit_transform(subset)

merged_df[columns_to_scale] = weather_features
merged_df = merged_df.dropna()
print(weather_features)

import math
# Teilen Sie die Daten in Trainings- und Testdaten auf
X_train, X_test, y_train, y_test = train_test_split(
    merged_df.drop(['Bike Count'], axis=1), 
    merged_df['Bike Count'], 
    test_size=0.2, 
    random_state=42
)

# Modelltraining
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Modelltest
predictions = model.predict(X_test)



import matplotlib.pyplot as plt

# Plotting the true values and predicted values along with the errors
plt.figure(figsize=(14, 6))

# Plotting the true values and predicted values as a line diagram
plt.plot(range(len(y_test)), y_test, color='blue', label='True Value')
plt.plot(range(len(predictions)), predictions, color='red', label='Predicted Value')

# Calculating the errors
errors = [abs(pred - y) for pred, y in zip(predictions, y_test)]

# Plotting the errors as a line within the same plot
plt.plot(range(len(predictions)), errors, color='green', label='Errors')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('True Value vs. Predicted Value with Errors')
plt.legend()
plt.show()
