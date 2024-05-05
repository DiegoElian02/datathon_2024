import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError, GeocoderTimedOut

# Función para realizar prediccion de pasajeros
def predict_passangers(data):    #aqui va el pickle
    return np.random.randint(100, 200)

with open(r'../models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open(r'../models/model.pkl', 'rb') as f:
    model = pickle.load(f)

ciudades_por_codigo = {
    'AT': 'Ciudad de Mexico', # México
    'BM': 'Tijuana',       # México
    'AW': 'Monterrey',         # México
    'BA': 'Merida',       # México
    'AJ': 'Tuxtla Gutierrez', # México
    'AO': 'Cancun',      # México
    'AK': 'Guadalajara', # México
    'BH': 'Saltillo',       # México
    'AY': 'Colima',         # México
    'BN': 'Victoria',       # México
    'AF': 'Durango',        # México
    'AU': 'Guanajuato',     # México
    'AD': 'Chilpancingo',   # México
    'BD': 'Pachuca',        # México
    'AR': 'Guadalajara',    # México
    'BJ': 'Toluca',         # México
    'BC': 'Morelia',        # México
    'BP': 'Cuernavaca',     # México
    'BG': 'Tepic',          # México
    'BL': 'Tel-Aviv',      # México
    'BQ': 'Oaxaca',         # México
    'AL': 'Puebla',         # México
    'AB': 'Queretaro',      # México
    'BF': 'Chetumal',       # México
    'BO': 'San Luis Potosi', # México
    'AP': 'Culiacan',       # México
    'BT': 'Hermosillo',     # México
    'BE': 'Villahermosa',   # México
    'BB': 'Ciudad Victoria', # México
    'AZ': 'Tlaxcala',       # México
    'AI': 'Xalapa',         # México
    'AQ': 'Mérida',         # México
    'BS': 'Los Angeles',    # EE.UU.
    'AX': 'New York',       # EE.UU.
    'AE': 'Chicago',        # EE.UU.
    'AV': 'Houston',        # EE.UU.
    'AM': 'Phoenix',        # EE.UU.
    'AS': 'Philadelphia',   # EE.UU.
    'BK': 'San Antonio',    # EE.UU.
    'BI': 'San Diego',      # EE.UU.
    'AC': 'Dallas'          # EE.UU.
}

def predict_products(data):
    data_df = pd.DataFrame([data], columns=['DepartureStation', 'ArrivalStation', 'Destination_Type', 'Origin_Type', 'Capacity', "Passengers", 'STD_hour', 'STA_hour', 'STD_day_of_week', 'STA_day_of_week'])
    data_transformed = preprocessor.transform(data_df)
    prediction = model.predict(data_transformed)
    return prediction 

# Configurar la página
st.set_page_config(layout="wide")

# Dividir la pantalla en dos columnas
col1, col2 = st.columns(2)

# Columna izquierda: entradas del usuario
with col1:
    st.header("Entrada para predicción de vuelo")
    departure_station = st.text_input('Departure Station')
    arrival_station = st.text_input('Arrival Station')
    destination_type = st.text_input('Destination Type')
    origin_type = st.text_input('Origin Type')
    capacity = st.number_input('Capacity', min_value=0)
    std_hour = st.number_input('STD Hour', min_value=0, max_value=23)
    sta_hour = st.number_input('STA Hour', min_value=0, max_value=23)
    std_day_of_week = st.number_input('STD Day of Week', min_value=0, max_value=6)
    sta_day_of_week = st.number_input('STA Day of Week', min_value=0, max_value=6)
    execute_button = st.button('Ejecutar Predicción')

def obtener_coordenadas(ciudad1, ciudad2):
    geolocator = Nominatim(user_agent="elianrdz2002@gmail.com")

    def geocode(ciudad):
        try:
            location = geolocator.geocode(ciudad)
            if location:
                return [location.longitude, location.latitude]
            else:
                return [None, None]
        except GeocoderTimedOut:
            print("La solicitud ha excedido el tiempo límite, intentando nuevamente...")
            return geocode(ciudad)  # Intenta nuevamente en caso de timeout
        except GeocoderServiceError as e:
            print(f"Error de servicio de geocodificación: {e}")
            return [None, None]

    coord_ciudad1 = geocode(ciudad1)
    coord_ciudad2 = geocode(ciudad2)

    return coord_ciudad1, coord_ciudad2
c1 = ciudades_por_codigo[departure_station]
c2 = ciudades_por_codigo[arrival_station]

coordenadas1, coordenadas2 = obtener_coordenadas(ciudades_por_codigo[departure_station], ciudades_por_codigo[arrival_station])

# Columna derecha: salida de predicciones
with col2:
    st.header("Resultados y Mapa del Vuelo")
    if execute_button:
        
        # Llamada a la función de predicción (ajusta los parámetros según sea necesario)
        features_pa = [departure_station, arrival_station, destination_type, origin_type, capacity, std_hour, sta_hour, std_day_of_week, sta_day_of_week]
        prediction_pa = predict_passangers(features_pa)
        st.subheader("Predicción de productos:")
        st.write(prediction_pa)
        features_pro = [departure_station, arrival_station, destination_type, origin_type, capacity, prediction_pa, std_hour, sta_hour, std_day_of_week, sta_day_of_week]
        prediction_pro = predict_products(features_pro)
        data_map = pd.DataFrame({
            'start': [coordenadas1],  # San Francisco
            'end': [coordenadas2]  # New York City
        })
        # Mapa (ajusta las coordenadas según necesidad)
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=coordenadas1[1],  # Usar el índice 1 para la latitud
                longitude=coordenadas1[0],  # Usar el índice 0 para la longitud
                zoom=4,  # Ajustar el zoom para ver ambos puntos
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'LineLayer',
                    data=data_map,
                    get_source_position='start',
                    get_target_position='end',
                    get_color=[255, 0, 0, 100],
                    get_width=5,
                ),
            ],
        ))
        st.write(departure_station)
        st.write(arrival_station)
        st.write(ciudades_por_codigo[departure_station])
        st.write(ciudades_por_codigo[arrival_station])
        st.subheader("Predicción de Productos:")
        st.write(prediction_pro)

# Nota: Asegúrate de tener las API Keys de mapbox configuradas si usas mapas con pydeck
