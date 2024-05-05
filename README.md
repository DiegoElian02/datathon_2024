# Proyecto Viva-Aerobus Datathon 2024

## Modelado de pasajeros

## Modelado de demanda de productos

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