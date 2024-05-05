import math
from statistics import mean
import streamlit as st
import pandas as pd
# import numpy as np
import pydeck as pdk
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.multioutput import MultiOutputRegressor
import pickle
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
# import os
# st.write(os.getcwd())

with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/pass_encoder.pkl', 'rb') as f:
    pass_encoder = pickle.load(f)

with open('models/pass_model.pkl', 'rb') as f:
    pass_model = pickle.load(f)

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

productos=['Agua Natural 600 Ml',
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
                'Xx Ultra']

means_products = [5.05977887e+00, 6.35690115e-01, 3.39614105e-02, 8.86785873e-02,
        5.41063804e-01, 4.28405803e-02, 6.37942898e-02, 2.40736726e-02,
        2.73350237e-04, 6.22324231e-01, 7.04046526e-01, 3.46410157e-01,
        4.03144470e-02, 4.47691133e-01, 6.31910341e-02, 3.37540413e-02,
        5.07017560e-02, 1.41388054e-04, 8.86031803e-04, 1.88517405e-05,
        1.15153029e+00, 3.62943134e-01, 7.85797099e-01, 7.53956509e-01,
        1.88517405e-05, 1.14566740e+00, 6.38995768e+00, 1.05586713e+00,
        1.10895363e-01, 1.37532873e-01, 5.50008955e-01, 7.54069619e-05,
        1.24336654e-01, 2.11411901e+00, 1.85133517e-01, 2.35891829e-01,
        6.89851166e-01, 1.48241604e+00, 7.43512645e-02, 5.56126344e-03,
        1.14524323e-02, 8.04026732e-03, 1.42849064e-01, 9.42587024e-06,
        1.69665664e-04, 5.88089470e-01, 2.08585083e-01, 1.77262916e-01,
        7.25273586e-01, 7.62043906e-01, 9.39287970e-02, 1.34723963e-01,
        9.42587024e-06, 2.45996362e-01, 2.33346844e-01, 2.59814687e-01,
        4.14738291e-04, 7.51053341e-02, 2.84764966e-01, 1.09528612e+00,
        9.42587024e-06, 1.21254395e-01, 3.09394765e-01, 1.99696487e-01,
        2.67317680e-01, 6.17564167e-01, 2.82776107e-05, 9.98520138e-01,
        6.18139145e-01, 7.24001093e-02, 1.03128446e-01, 1.65305257e+00,
        1.06731014e+00, 2.86075162e-02, 9.95560415e-02, 3.77034810e-05,
        7.77351519e-02, 4.71293512e-05, 3.61764900e-01, 1.18294672e-01,
        2.21778473e+00, 1.37366035e+00, 3.01114138e+00, 6.95629224e-03,
        8.72939269e-01, 2.05785599e-01, 1.31421138e+00, 6.95534965e-02,
        1.00790831e-01, 1.65329764e-02, 2.67506198e-02, 6.10419357e-01,
        1.67045272e-01, 1.84973278e-01, 1.43895335e-01, 8.25423457e-02,
        2.33101771e-02, 1.62379467e-01, 6.17799813e-01, 7.16743173e-02,
        2.90693838e-02, 3.07848922e-02, 5.92038910e-02, 2.23496809e-01,
        6.00314824e-01, 3.24683526e-01]

# Definición de la función is_high_season
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

# Función para preparar los datos de entrada para pass pred
def prepare_flight_data(input_data):
    # Desempaquetar la entrada
    (departure_station, arrival_station, destination_type, origin_type, capacity, 
     std_month, std_day, std_year, std_hour, sta_month, sta_day, sta_year, sta_hour) = input_data
    
    # Crear las fechas de STD y STA para usar en cálculos
    std_date = pd.Timestamp(year=std_year, month=std_month, day=std_day)
    # sta_date = pd.Timestamp(year=sta_year, month=sta_month, day=sta_day)
    
    # Calcular si es temporada alta y si es fin de semana
    high_season = is_high_season(std_date)
    is_weekend = 1 if std_date.dayofweek >= 5 else 0
    
    # Preparar el diccionario de salida
    output_data = {
        'DepartureStation': departure_station,
        'ArrivalStation': arrival_station,
        'Destination_Type': destination_type,
        'Origin_Type': origin_type,
        'Capacity': capacity,
        'STD_Month': std_month,
        'STD_Hour': std_hour,
        'STA_Year': sta_year,
        'STA_Month': sta_month,
        'STA_Day': sta_day,
        'STA_Hour': sta_hour,
        'high_season': high_season,
        'Is_Weekend': is_weekend
    }
    return output_data

def predict_products(data):
    data_df = pd.DataFrame([data], columns=['DepartureStation', 'ArrivalStation', 'Destination_Type', 'Origin_Type', 'Capacity', "Passengers", 'STD_hour', 'STA_hour', 'STD_day_of_week', 'STA_day_of_week'])
    data_transformed = preprocessor.transform(data_df)
    prediction = model.predict(data_transformed)
    return prediction 

def predict_passangers(data):    #aqui va el pickle
    # columns = ['DepartureStatio', 'ArrivalStation', 'Destination_Type', 'Origin_Type', 
    #            'Capacity', "STD_Month", 'STD_Hour', "STA_Year", "STA_Month", "STA_Day", 'STA_Hour', "high_season", "Is_Weekend"]
    input_df = pd.DataFrame(data, index=[0])
    # Aplicar el codificador a los datos de entrada
    input_encoded = pass_encoder.transform(input_df)
    
    # Hacer predicciones
    predicted_passengers = pass_model.predict(input_encoded)
    return round(predicted_passengers[0])

# Configurar la página
st.set_page_config(layout="wide")

# Dividir la pantalla en dos columnas
col1, col2 = st.columns(2)

# Columna izquierda: entradas del usuario
with col1:
    st.header("Entrada para predicción de vuelo")
    departure_station = st.selectbox('Departure Station', list(ciudades_por_codigo.keys()))
    arrival_station = st.selectbox('Arrival Station', list(ciudades_por_codigo.keys()))
    destination_type = st.selectbox('Destination Type', ['Ciudad Principal', 'Ecoturismo', 'Playa', 'Ciudad Fronteriza', 'MX Amigos y Familia'])
    origin_type = st.selectbox('Origin Type', ['Ciudad Principal', 'Ecoturismo', 'Playa', 'Ciudad Fronteriza', 'MX Amigos y Familia'])
    capacity = st.number_input('Capacity', min_value=0)
    std_month = st.number_input('STD Month', min_value=1, max_value=12)
    std_day = st.number_input('STD Day', min_value=1, max_value=31)
    std_year = st.number_input('STD Year', min_value=2023, max_value=2024)
    std_hour = st.number_input('STD Hour', min_value=0, max_value=23)
    sta_month = st.number_input('STA Month', min_value=1, max_value=12)
    sta_day = st.number_input('STA Day', min_value=1, max_value=31)
    sta_year = st.number_input('STA Year', min_value=2023, max_value=2024)
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


# Columna derecha: salida de predicciones
with col2:
    st.header("Resultados y Mapa del Vuelo")
    if execute_button:
        c1 = ciudades_por_codigo[departure_station]
        c2 = ciudades_por_codigo[arrival_station]
        coordenadas1, coordenadas2 = obtener_coordenadas(ciudades_por_codigo[departure_station], ciudades_por_codigo[arrival_station])

        # Llamada a la función de predicción (ajusta los parámetros según sea necesario)
        features_pa = [departure_station, arrival_station, destination_type, origin_type, capacity, std_month, std_day, std_year, std_hour, sta_month, sta_day, sta_year, sta_hour]
        process_for_pass_pred = prepare_flight_data(features_pa)
        prediction_pa = predict_passangers(process_for_pass_pred)
        st.subheader("Predicción de pasajeros:")
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
        
        st.subheader("Predicción de Productos:")
        prediction_pro_with_mean = [x + y for x, y in zip(prediction_pro, means_products)]
        st.write(pd.DataFrame(prediction_pro_with_mean, columns=productos))
    df_opti = pd.read_parquet('data/Results_OptimalBoxDistribution.parquet')
    df_html = df_opti.to_html(index=True)

    # Crear un contenedor scrolleable usando HTML/CSS
    scrollable_container = f"""
    <div style="height:460px;overflow-y:scroll;border:1px solid #e6e9ef;border-radius:0.25rem;padding:1rem;">
        {df_html}
    </div>
    """
    st.markdown(scrollable_container, unsafe_allow_html=True)

# Nota: Asegúrate de tener las API Keys de mapbox configuradas si usas mapas con pydeck