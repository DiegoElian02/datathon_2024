import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# pd.set_option('display.max_columns', 500)

# Load the data
flights_with_products = pd.read_parquet("../data/flights_with_products.parquet")
# flights_with_products.head()

flights_data = flights_with_products.dropna()
flights_data = flights_data.reset_index(drop=True)

# Convertir las columnas de fecha y hora
flights_data['STD'] = pd.to_datetime(flights_data['STD'])
flights_data['STA'] = pd.to_datetime(flights_data['STA'])

# Extraer características de fecha y hora
flights_data['STD_hour'] = flights_data['STD'].dt.hour
flights_data['STA_hour'] = flights_data['STA'].dt.hour
flights_data['STD_day_of_week'] = flights_data['STD'].dt.dayofweek
flights_data['STA_day_of_week'] = flights_data['STA'].dt.dayofweek

x_columns = ['Aeronave', 'DepartureStation', 'ArrivalStation', 'Destination_Type', 'Origin_Type', 'Capacity', 'Passengers', 'STD_hour', 'STA_hour', 'STD_day_of_week', 'STA_day_of_week']
y_columns = [col for col in flights_data.columns if col not in x_columns + ['Flight_ID'] + ['STD'] + ['STA'] + ['Bookings']]


X = flights_data[x_columns]
y = flights_data[y_columns]

y = y[['Agua Natural 600 Ml',
 'Amstel Ultra',
 'Arandano',
 'Arandano Mango Mix',
 'Arcoiris',
 'Baileys',
 'Baileys ',
 'Botana Sabritas Con Dip De Queso',
 'Cafe 19 Cafe Clasico',
 'Cafe 19 Capuchino',
 'Cafe 19 Chiapas',
 'Cafe Costa',
 'Cafe De Olla',
 'Capitan Morning',
 'Capitan Morning Con Pan Dulce',
 'Carne Seca Habanero',
 'Carne Seca Original',
 'Cerveza Charter',
 'Charter Cheve Doble',
 'Charter Licor Doble',
 'Cheetos',
 'Cheetos Flamin Hot',
 'Chokis',
 'Ciel Mineralizada',
 'Club Sandwich',
 'Coca Cola Dieta',
 'Coca Cola Regular',
 'Coca Sin Azucar',
 'Corajillo',
 'Corajillo Baileys ',
 'Cuerno Clasico De Pavo',
 'Cuerno Individual Charter',
 'Dip De Queso',
 'Doritos Nacho',
 'Emperador Chocolate',
 'Emperador Vainilla',
 'Fanta De Naranja',
 'Fritos Limon Y Sal',
 'Frutos Secos Enchilados',
 'Galleta De Arandano Relleno De Q/Crema',
 'Galleta De Chispas De Chocolate',
 'Galleta De Chocolate',
 'Go Nuts',
 'Gomita Enchilada La Cueva',
 'Heineken 0',
 'Heineken Original',
 'Heineken Silver',
 'Jack And Coke',
 'Jugo De Mango',
 'Jugo De Manzana',
 'Jw Red Label',
 'Jw Red Label ',
 'Kacang Flaming Hot',
 'Leche De Chocolate Sc',
 'Leche De Fresa Sc',
 'Licor + Refresco',
 'Licor Charter',
 'Luxury Nut Mix',
 'Mafer Sin Sal',
 'Mega Cuerno Clasico',
 'Mega Cuerno Tripulacion',
 'Muffin Integral',
 'Nishikawa Japones',
 'Nishikawa Salado',
 'Nissin Dark Dragon',
 'Nissin Fuego',
 'Nissin Limon Y Habanero',
 'Nissin Picante',
 'Nissin Res',
 'Nueces De Arbol Mix',
 'Nutty Berry Mix',
 'Panini Clasico',
 'Panini Integral',
 'Protein Adventure',
 'Quaker Avena Frutos Rojos',
 'Quaker Avena Moras',
 'Quaker Granola',
 'Quaker Natural Balance',
 'Rancheritos',
 'Ron Bacardi',
 'Ruffles Queso',
 'Sabritas Flamin Hot',
 'Sabritas Originales',
 'Salsa Botanera',
 'Sidral Mundet',
 'Sol Clamato',
 'Sprite',
 'Te Frutos Rojos',
 'Te Manzanilla Jengibre',
 'Te Relax',
 'Te Vainilla',
 'Tecate Light',
 'Tequila + Mezclador',
 'Tequila 7 Leguas Blanco',
 'Tequila 7 Leguas Reposado',
 'Tinto',
 'Topochico Seltzer Fresa-Guayaba',
 'Topochico Seltzer Mango',
 'Tostitos',
 'Tostitos Nachos Con Dip',
 'Ultra Seltzer Frambuesa',
 'Vino Blanco Cria Cuervos ',
 'Vino Tinto Cria Cuervos',
 'Vino Tinto Sangre De Toro',
 'Xx Lager',
 'Xx Ultra']]

# Preprocesamiento
categorical_features = ['DepartureStation', 'ArrivalStation', 'Destination_Type', 'Origin_Type']
numeric_features = ['Capacity', 'Passengers', 'STD_hour', 'STA_hour', 'STD_day_of_week', 'STA_day_of_week']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Preparar datos
X_transformed = preprocessor.fit_transform(X)
y_array = y.values

# Guardar el preprocessor
with open(r'../models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_array, test_size=0.2, random_state=0)

# Definición del modelo de red neuronal
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation="relu")  # Número de neuronas , de salida igual al número de productos
])

model.compile(optimizer="Adam", loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Realiza las predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Guardar el modelo
with open(r'../models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Calcula las métricas de error
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir las métricas
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)