import numpy as np

# Supongamos que tienes las predicciones de cada modelo en una lista de listas
predicciones = [
    [0, 1, 0, 0, 1],  # Predicciones del Modelo 1
    [1, 0, 1, 0, 0],  # Predicciones del Modelo 2
    [0, 1, 1, 0, 1],  # Predicciones del Modelo 3
    [1, 1, 1, 1, 1],  # Predicciones del Modelo 4
    [0, 0, 1, 1, 0],  # Predicciones del Modelo 5
]

# Pesos para cada modelo (el Modelo 4 tiene el peso más alto)
pesos = [0.15, 0.15, 0.15, 0.30, 0.15]  # Suma de pesos = 1

# Calcular la predicción final con promedio ponderado
prediccion_final = np.average(predicciones, axis=0, weights=pesos)

# Convertir a valores binarios (0 o 1)
prediccion_final = np.where(prediccion_final >= 0.5, 1, 0)

print("Predicción final:", prediccion_final)