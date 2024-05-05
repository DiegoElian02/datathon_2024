import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

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

x_columns = ['Aeronave', 'DepartureStation', 'ArrivalStation', 'Destination_Type', 'Origin_Type', 'Capacity', 'Passengers', 'Bookings', 'STD_hour', 'STA_hour', 'STD_day_of_week', 'STA_day_of_week']
y_columns = [col for col in flights_data.columns if col not in x_columns + ['Flight_ID'] + ['STD'] + ['STA']]


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

categorical_features = ['Aeronave', 'DepartureStation', 'ArrivalStation', 'Destination_Type', 'Origin_Type']
numeric_features = ['Capacity', 'Passengers', 'Bookings', 'STD_hour', 'STA_hour', 'STD_day_of_week', 'STA_day_of_week']

# Crear un transformador de columnas para aplicar OneHotEncoding a las categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Preparar el pipeline con el preprocesador y el modelo
model = MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', n_estimators=100, verbosity=2, n_jobs=-1))

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
pipeline.fit(X_train, y_train)

# Evaluación del modelo (opcional)
score = pipeline.score(X_test, y_test)
print("Score del modelo: ", score)