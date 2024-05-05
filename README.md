# Proyecto Viva-Aerobus Datathon 2024

## Modelado de pasajeros

Se ha establecido el objetivo de predecir el volumen de pasajeros dada una base de datos que contiene la siguiente información de diferentes vuelos:

1) Flight_ID: ID único de cada vuelo.\\

2) Aeronave: ID único por aeronave.\\

3) DepartureStation: Aeropuerto de salida del vuelo.\\

4) ArrivalStation: Aeroupuerto de llegada del vuelo.\\

5) Destination_Type / Origin_Type: Tipo de destino y de origen de cada vuelo.\\

6) STD: Timestamp de salida del vuelo.\\

7) STA: Timestamp de aterrizaje del vuelo.\\

8) Capacity: Capacidad real de pasajeros del avión.\\

9) Passengers: Cantidad de personas que abordaron el vuelo.\\

10) Bookings: Total de reservas del vuelo.\\

Es importante tener en consideración que se tiene información de vuelos a partir del 2 de Enero de 2023 hasta algunas fechas ya reservadas para 2025, sin embargo, únicamente se tiene información de la cantidad de pasajeros y bookings de vuelos de 2023, mientras que el resto están dados por NaN.

Para limpiar la base de datos fue necesario deshacerse de los NaN, por lo que se ha realizado un conteo total de los registros nulos dentro del dataframe. Se obtuvo que existen 80,000 registros nulos dentro de la variable 'Aeronave', mientras que habían 123,500 registros nulos en las variables de 'Passengers' y 'Bookings'. También hubo un solo registro nulo en columnas como 'DepartureStation', 'ArrivalStation', 'Destination_Type' y 'Origin_Type'.

Para fines de este modelo, se han eliminado las columnas 'Flight_ID', 'Aeronave' y 'Bookings', ya que las primeras dos no representan información importante que el algoritmo deba considerar a la hora de tomar una decisión (la característica importante de la aeronave es la capacidad y está dada en una columna diferente). Por su parte, la variable 'Passengers' ya contiene los clientes que realizaron alguna reserva, por lo que predecir esta variable hace despreciable la columna 'Bookings'. Este proceso ha hecho que los registros nulos desaparezcan, dejando únicamente las columnas con un valor nulo en ellas. Para estos casos, se han eliminado las filas totales (2) que seguían contando con NaN.

Las siguientes variables que se modificaron para el modelo fueron 'STD' y 'STA'. Se crearon columnas adicionales en las que se separaron la fecha de despegue ('Date_STD'), la hora de despegue ('Time_STD'), la fecha de aterrizaje ('Date_STA') y la hora de aterrizaje ('Time_STA'). Además, las horas de despegue y aterrizaje han sido redondeadas para facilitar el entrenamiento del algoritmo. De igual manera, se ha agregado una columna que indique si la fecha del año es temporada alta o no, considerando las vacaciones de verano, las vacaciones de invierno, semana santa y algunos asuetos, todo con base en el calendario de la SEP de México.

Las columnas de fecha de aterrizaje y fecha de despegue se han modificado para que únicamente muestren el número de mes en el que se realizó el viaje, dado que solamente se tiene información de un año, lo que hace que la información del año del vuelo sea despreciable, al igual que el día específico (solamente se estudia una vez cada día del año por lo que no es posible encontrar patrones complejos).

Finalmente, para tener la información lista para comenzar a trabajar con los modelos de predicción, se hizo un duplicado de la base de datos creando columnas 'dummies' para las variables de 'DepartureStation', 'ArrivalStation', 'Destination_Type' y 'Origin_Type', de manera que algunos de los algoritmos de regresión pudieran leer la información de una manera más sencilla. Por su parte, las columnas que contenían la hora de despegue y la hora de aterrizaje se modificaron para que únicamente contuvieran el número de hora (0-23). Se creó un csv adicional con ambas bases de datos (con 'dummies' y sin ellos) para facilitar la lectura en el futuro.

## Modelado de demanda de productos

### Descripción del Proceso
El objetivo del modelo es predecir la cantidad de productos vendidos en vuelos, basándose en características del vuelo como la estación de salida, la estación de llegada, el tipo de destino, entre otros.

### Preparación de los Datos

1. **Carga de Datos:** Los datos se cargan desde un archivo Parquet que contiene tanto las características de los vuelos como la cantidad de productos vendidos.
2. **Limpieza de Datos:** Se eliminan los valores faltantes y se reinicia el índice del DataFrame.
3. **Transformación de Fechas:** Las columnas de fecha y hora son convertidas a `datetime`, extrayendo la hora y el día de la semana para la salida (`STD`) y llegada (`STA`) del vuelo.

### Preprocesamiento de Datos

- **Codificación y Escalado:** Las características categóricas son codificadas utilizando `OneHotEncoder`, y las características numéricas son escaladas usando `StandardScaler`.
- **Transformación:** Se aplica la transformación en conjunto a los datos de entrada (`X`), preparándolos para el modelo.

### Creación y Entrenamiento del Modelo

- **Arquitectura:** Se utiliza un modelo de red neuronal profunda con dos capas ocultas de 128 y 64 nodos, regularización L2 y Dropout para evitar el sobreajuste.
- **Compilación:** El modelo se compila con el optimizador `Adam` y la función de pérdida de error cuadrático medio (MSE).
- **Entrenamiento:** El modelo se entrena con los datos, utilizando un conjunto de validación para monitorear el rendimiento durante el entrenamiento.

### Evaluación del Modelo

- **Predicción:** Se realizan predicciones en el conjunto de prueba para evaluar la capacidad del modelo de generalizar a nuevos datos.
- **Métricas de Rendimiento:** Se calculan el Error Cuadrático Medio (MSE), Error Absoluto Medio (MAE) y el coeficiente de determinación (R²) para medir la precisión y efectividad del modelo.

## Modelo Optimizador

Una vez tenemos nuestra predicción de la demanda podemos generar un modelo para optimizar la distribución de los reabastecimientos. Como primera instancia generamos la información de manera que tenemos guardado la ultima vez que cada avión aterrizó en un aeropuerto apto para abastecerse y para este modelo agrupamos por esta nueva variable. Es decir que ahora si un avión sale de Monterrey hacia Chihuahua y luego a CDMX vamos a considerar que desde Monterrey debe de traer tambien los productos para el vuelo desde Chihuahua. 

Teniendo esto en cuenta, haremos una optimización multiobjetivo en donde

$$
\text{Sea     }R_v = \begin{cases} 
1 & \text{Si el vuelo $v$ se reabastece de productos}, \\
0 & \text{De otra forma}.
\end{cases}
$$

- Sea $C_{t,v}$ la cantidad de cajas del tipo $t$ que se agregan antes del vuelo $v$. 

Ahora tenemos 2 funciones objetivo. La primera se enfoca en minimizar el desperdicio o sobras de los aviones y la segunda minimiza la cantidad de veces que se tiene que reabastecer un avion. 

Minimizamos la diferencia que hay entre cada producto abastecido en el avion y su demanda. 

1) $$\text{Min      } \sum_v \sum_p ProductoEnCaja_{p,t} * C_{t,v} - Demanda_{p,v}$$


Minimizamos la cantidad de veces que se rabastecen los aviones. 

2) $$\text{Min      } \sum_v R_{v}$$

Ahora tenemos que este modelo debe estar sujeto a las restricciones:

- No se puede cargar cajas en aviones si no se agrega la variable de reabastecimiento. 
$$R_v * 2^{30} \geq \sum_t C_{t,v} \quad \quad \forall\ v$$

- Nuestra demanda debe ser satisfecha siempre.
$$ProductoEnCaja_{p,t} * C_{t,v} \geq Demanda_{p,v} \quad \quad \forall \ p,v$$

- Tenemos un límite de cajas que podemos agregar al avión.
$$\sum_t C_{t,v} \leq LimiteCajas$$

Cabe recalcar que se utiliza un metodo lexicográfico para la solución multiobjetivo en donde nuestra primera función objetivo no puede desviarse más del 5% al hacer la segunda optimización. 