# Import libraries
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle

# Import dataset
df_flights = pd.read_csv("../data/Filghts TEC_Valid.csv")

# Separate the STD AND STA column into year, month, day and time. Rounding the hour column

# Convert STD and STA columns to datetime
df_flights['STD'] = pd.to_datetime(df_flights['STD'])
df_flights['STA'] = pd.to_datetime(df_flights['STA'])

# Extract year, month, day and time from STD
df_flights['STD_Year'] = df_flights['STD'].dt.year
df_flights['STD_Month'] = df_flights['STD'].dt.month
df_flights['STD_Day'] = df_flights['STD'].dt.day
df_flights['STD_Hour'] = df_flights['STD'].dt.hour + round(df_flights['STD'].dt.minute / 60)

# Extract year, month, day and time from STA
df_flights['STA_Year'] = df_flights['STA'].dt.year
df_flights['STA_Month'] = df_flights['STA'].dt.month
df_flights['STA_Day'] = df_flights['STA'].dt.day
df_flights['STA_Hour'] = df_flights['STA'].dt.hour + round(df_flights['STA'].dt.minute / 60)

# Function to determine if the date is high season
def is_high_season(date):
    year = date.year
    if (pd.Timestamp(year=year, month=7, day=17) <= date <= pd.Timestamp(year=year, month=8, day=27)) or \
       (pd.Timestamp(year=year, month=11, day=18) <= date <= pd.Timestamp(year=year, month=11, day=20)) or \
       (pd.Timestamp(year=year, month=12, day=18) <= date <= pd.Timestamp(year=year+1, month=1, day=5)) or \
       (pd.Timestamp(year=year, month=2, day=3) <= date <= pd.Timestamp(year=year, month=2, day=5)) or \
       (pd.Timestamp(year=year, month=3, day=16) <= date <= pd.Timestamp(year=year, month=3, day=18)):
        return 1
    if year == 2024 and (pd.Timestamp(year=2024, month=3, day=25) <= date <= pd.Timestamp(year=2024, month=4, day=7)):
        return 1
    if year == 2023 and (pd.Timestamp(year=2023, month=4, day=1) <= date <= pd.Timestamp(year=2023, month=4, day=16)):
        return 1
    return 0

# Apply the function to the STD column to create a new column 'high_Season'
df_flights['high_season'] = df_flights['STD'].apply(is_high_season)

# df_flights

df_flights['Is_Weekend'] = df_flights['STD'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

df_flights2023 = df_flights[df_flights['STD_Year'] == 2023]

X = df_flights2023.drop(columns = ['Passengers', 'Flight_ID','Aeronave', 'STD', 'STA', 'Bookings', 'STD_Year', 'STD_Day' ] , axis=1)
y = df_flights2023['Passengers']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the encoder by specifying the categorical columns
encoder = ce.TargetEncoder(cols=['DepartureStation', 'ArrivalStation', 'Destination_Type', 'Origin_Type'])

# Tune the encoder with training data
encoder.fit(X_train, y_train)

# Transform the data
X_train_encoded = encoder.transform(X_train)
X_test_encoded = encoder.transform(X_test)

n_estimators = 300 
max_depth = 30     
min_samples_split = 5

rf_model  = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split = min_samples_split, verbose=2, n_jobs=-1)

rf_model.fit(X_train_encoded, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test_encoded)

# Calculate and display R^2
r2 = r2_score(y_test, y_pred)
print(f"Determination Coefficient (R^2): {r2}")

# Guardar el modelo
with open(r'../models/pass_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open(r'../models/pass_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)