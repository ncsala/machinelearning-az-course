# -*- coding: utf-8 -*-
"""
Aqui simplemente pre procesamos los datos. Los cargamos a una variable dataset
y luego definimos las variables q vamos a usar como independientes y dependientes
"""

import numpy as np
import matplotlib.pyplot
import pandas as pd

# Importar el DataSet
dataset = pd.read_csv('Data.csv')

# Le digo que tome las tres primeras columnas 
# y todas las filas del dataset como variable x.
# '.values' -> significa que quiero extraer solo valores y no posiciones.
# Estas son las variables independientes que vamos a tomar para predecir.
# con los dos puntos -1 seleccionamos todas las filas y todas las columnas
# menos la ultima.
x = dataset.iloc[:, :-1].values 
# Ahora creamos la variable dependiente, la que queremos predecir
# Seleccionamos sola la ultima columna
y = dataset.iloc[:, 3]

# Tratamiento de las NAS.
# O sea cuando importamos datos muchos muchos vienen incompletos, tenemos que tratarlos
# de alguna manera, una forma por ejemplo es insertar un dato promedio de la fila.
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])