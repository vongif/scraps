import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

millas = np.array([40, 10, 0, 8, 15, 22, 38, 50], dtype=float)
kilometros = np.array([40, 14, 32, 46, 59, 72, 100, 80], dtype=float)

#capa = tf.keras.layers.Dense(units=1, input_shape=[1])
#modelo = tf.keras.Sequential([capa])
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=1)
oculta3 = tf.keras.layers.Dense(units=1)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, oculta3, salida])


modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(millas, kilometros, epochs=2000, verbose=False)
print("Modelo entrenado!!")


plt.xlabel("# Epoca")
plt.ylabel("Magniud de perdida")
plt.plot(historial.history["loss"])

print("Hagamos una prediccion!")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + "kilometros!")

print("Variables internas del modelo")
#print(capa.get_weights())
print(oculta1.get_weights())
print(oculta2.get_weights())
print(oculta3.get_weights())
print(salida.get_weights())