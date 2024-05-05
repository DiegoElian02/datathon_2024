#%%
import pandas as pd
import pulp as pl
import itertools
import numpy as np
import openpyxl

#%% Empezamos la optimizacion multiobjetivo declarando las variables de decision
prob = pl.LpProblem("LexicographicOptimization", pl.LpMinimize)

#Parametros
tipos = [1, 2]
vuelos = [1,2,3]
LimiteCajas = 15
productos = ['Chocolate', 'Zanahoria', 'Vainilla']

Demanda = pd.read_csv(r'C:\Users\elias\OneDrive\Desktop\datathon_2024\data\Dummy\Demanda.csv').set_index(['Vuelo', 'producto'])
ProductoXCaja = pd.read_csv(r'C:\Users\elias\OneDrive\Desktop\datathon_2024\data\Dummy\ProductosXCaja.csv').set_index(['producto', 'tipo'])

#Variables de decision
C = pl.LpVariable.dicts('C', list(itertools.product(tipos, vuelos)), lowBound=0, upBound=20, cat='Integer')

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