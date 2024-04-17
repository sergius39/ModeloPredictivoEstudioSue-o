
import tensorflow as tf
import numpy as np

#Valores
horas_estudiadas = np.array([0, 1, 2, 3, 4, 5], dtype=float)
horas_dormidas = np.array([8, 7, 6, 5, 4, 3], dtype=float)
notas_examen = np.array([2, 4, 5, 6, 7, 9], dtype=float)
niveles_energia = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.2], dtype=float)

#Capas
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation='relu', input_shape=[1], name='horas_estudiadas'),
    tf.keras.layers.Dense(units=5, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='relu', input_shape=[1], name='horas_dormidas'),
    tf.keras.layers.Dense(units=5, activation='relu'),
    tf.keras.layers.Dense(units=1, name='notas_examen'),
    tf.keras.layers.Dense(units=1, name='niveles_energia')
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

#Entrenamiento
print("===========================")
print("Comenzando entrenamiento...")
print("===========================")

modelo.fit([horas_estudiadas,horas_dormidas], [notas_examen, niveles_energia], epochs=1000, verbose=False)

print("=================")
print("Modelo entrenado!")
print("=================")

# Prediccion
print("Hagamos una prediccion!")
entrada_prediccion = np.array([[1], [7]])
nota, energia = modelo.predict(entrada_prediccion)

print("==================================================")
print("Nota:", nota)
print("Energia:", energia)
print("==================================================")