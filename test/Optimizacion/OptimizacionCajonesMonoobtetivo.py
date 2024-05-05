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

LimiteCajas = 5000
productos = ['Agua Natural 600 Ml',
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
productos = [p.upper() for p in productos]
ProductoXCaja = pd.read_csv(r'..\data\Tablasoptimizacion.csv')
ProductoXCaja = ProductoXCaja[ProductoXCaja['producto'].isin(productos)]
productosEnCajas = ProductoXCaja['producto'].unique()
productos = productosEnCajas

all_combinations = pd.MultiIndex.from_product([productosEnCajas, tipos], names=['producto', 'tipo'])
ProductoXCaja = ProductoXCaja.set_index(['producto', 'tipo'])
ProductoXCaja = ProductoXCaja.reindex(all_combinations, fill_value=0)

Demanda = pd.melt(flight_df.drop(columns=['Aeronave', 'Capacity', 'Passengers', 'Bookings']),
                  id_vars=['LastLoadChance'],
                  var_name='producto')
Demanda = Demanda[Demanda['producto'].isin(productos)]
Demanda = Demanda.set_index(['LastLoadChance', 'producto'])

#Variables de decision
C = pl.LpVariable.dicts('C', list(itertools.product(tipos, vuelos)), lowBound=0, upBound=4)

#%% Restricciones
    
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

#%% Añadiendo optimizacion multiobjetivo

primary_obj = pl.value(prob.objective)

prob.sense = pl.LpMinimize

#Añadir que solo podamos deviar 5% del maximo ingreso
prob += pl.lpSum(Ingresos)
        
prob.setObjective(pl.lpSum([Demanda.loc[(p,c,t)][0] for (p,c,t) in list(itertools.product(productos, clientes, meses))])\
                  - pl.lpSum(Z[(p,f,c,t)] for (p,f,c,t) in list(itertools.product(productos, fabricas, clientes, meses)))
                )

prob.solve()