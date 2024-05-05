#%%
import pandas as pd
import pulp as pl
import itertools
import numpy as np
import openpyxl

#%% Procesando vuelos para agrupar hasta que lleguen a un aeropuerto apto para reabastecerse

flight_df = pd.read_csv(r'..\data\Dummy_results_Jan01.csv').drop(columns=['Unnamed: 0'])
flight_df['LastLoadChance'] = flight_df['DepartureStation']

#TODO: Esto es una fuerza bruta pero funciona y es muy tarde para optimizarla
#Generando la columna 'LastLoadChance' con el ultimo aeropuerto al cual se puede cargar cajas
aeropuertos = [ 'AW',    #'Monterrey',
                'AT',    #'AICM',
                'AO',    #'Cancún',
                'AK',    #'Guadalajara',
                'BM',    #'Tijuana',
                'BA',    #'Mérida',
                'AU']   #'Guanajuato' 

flight_dict = dict()
for a in flight_df['Aeronave'].unique():
    itinerario = flight_df[flight_df['Aeronave'] == a]
    itinerario = itinerario.sort_values(by='STD').reset_index(drop='true')
    
    last_important_port = 'overnight'
    
    for i in range(len(itinerario)):
        if(itinerario.loc[i]['DepartureStation'] in (aeropuertos)):
            last_important_port = itinerario.loc[i]['DepartureStation'] + ' - ' + itinerario.loc[i]['Flight_ID']

        itinerario.loc[i, 'LastLoadChance'] = last_important_port
    
    flight_dict[a] = itinerario


flight_df = pd.concat([flight_dict[a] for a in flight_df['Aeronave'].unique()], ignore_index=True)

#Agrupando dataframe por 'LastLoadChance' y nave
flight_df = flight_df.groupby(by=['Aeronave', 'LastLoadChance']).sum(numeric_only=True).reset_index()

#%% Empezamos la optimizacion multiobjetivo declarando las variables de decision
prob = pl.LpProblem("LexicographicOptimization", pl.LpMinimize)

#Parametros
tipos = list(range(1,29))
vuelos = flight_df['LastLoadChance'].unique()
LimiteCajas = 1000

ProductoXCaja = pd.read_csv(r'..\data\Tablasoptimizacion.csv')

Demanda = pd.melt(flight_df.drop(columns=['Aeronave', 'Capacity', 'Passengers', 'Bookings']),
                  id_vars=['LastLoadChance'],
                  var_name='producto')

productos = list(set(Demanda['producto']).intersection(set(ProductoXCaja['producto'])))

#filtrando datos faltantes
Demanda = Demanda[Demanda['producto'].isin(productos)]
ProductoXCaja = ProductoXCaja[ProductoXCaja['producto'].isin(productos)]

all_combinations = pd.MultiIndex.from_product([productos, tipos], names=['producto', 'tipo'])
ProductoXCaja = ProductoXCaja.set_index(['producto', 'tipo'])
ProductoXCaja = ProductoXCaja.reindex(all_combinations, fill_value=0)
Demanda = Demanda.set_index(['LastLoadChance', 'producto'])

#%% Restricciones

#Variables de decision
C = pl.LpVariable.dicts('C', list(itertools.product(tipos, vuelos)), lowBound=0, upBound=1000, cat='Integer')
    
#* Satisfacemos la demanda estimada
for (p,v) in list(itertools.product(productos, vuelos)):
    prob += pl.lpSum([ProductoXCaja.loc[p, t][0] * C[(t, v)] for t in tipos]) >= Demanda.loc[(v, p)]['value']

#* No ingresamos más cajas de las que caben en el avion
for v in vuelos:
    prob += pl.lpSum([C[(t,v)] for t in tipos]) <= LimiteCajas

#%% Funcion Objetivo 1: 

prob += pl.lpSum([ProductoXCaja.loc[p,t][0] * C[(t,v)] for (p,t,v) in list(itertools.product(productos, tipos, vuelos))])\
        - pl.lpSum([Demanda.loc[v, p]['value'] for p,v in list(itertools.product(productos, vuelos))])
#%%
prob.solve()
