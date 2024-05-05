import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Función para realizar prediccion de pasajeros
def predict_passangers(aeronave, departure_station, arrival_station, destination_type, origin_type, capacity, std_hour, sta_hour, std_day_of_week, sta_day_of_week):
    #aqui va el pickle
    return np.random.randint(100, 200)

def predict_products(aeronave, departure_station, arrival_station, destination_type, origin_type, capacity, std_hour, sta_hour, std_day_of_week, sta_day_of_week):
    
    return 


# Configurar la página
st.set_page_config(layout="wide")

# Dividir la pantalla en dos columnas
col1, col2 = st.columns(2)

# Columna izquierda: entradas del usuario
with col1:
    st.header("Entrada para predicción de vuelo")
    aeronave = st.text_input('Aeronave')
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

# Columna derecha: salida de predicciones
with col2:
    st.header("Resultados y Mapa del Vuelo")
    if execute_button:
        # Llamada a la función de predicción (ajusta los parámetros según sea necesario)
        features = np.array([[aeronave, departure_station, arrival_station, destination_type, origin_type, capacity, std_hour, sta_hour, std_day_of_week, sta_day_of_week]])
        prediction_pa = predict_passangers(features)
        prediction_pro = predict_products(features)
        # Mostrar predicciones
        st.subheader("Predicción de Pasajeros:")
        st.write(prediction_pa)
        
        # Mapa (ajusta las coordenadas según necesidad)
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=37.7749,
                longitude=-122.4194,
                zoom=11,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                   'LineLayer',
                   data=pd.DataFrame({
                       'start': [[-122.4194, 37.7749]],
                       'end': [[-73.935242, 40.730610]]
                   }),
                   get_source_position='start',
                   get_target_position='end',
                   get_color=[255, 0, 0, 100],
                   get_width=5,
                ),
            ],
        ))

        st.subheader("Predicción de Productos:")
        st.write(prediction_pro)

# Nota: Asegúrate de tener las API Keys de mapbox configuradas si usas mapas con pydeck
