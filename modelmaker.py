import tensorflow as tf
import numpy as np
import cv2 as cv

#config
model_name = 'model'
EPOCHS = 30
imageres = 64

#load data
x = np.load("images.npy")
y = np.load("labels.npy")
print(x.shape)
print(y.shape)


#create model
model = tf.keras.models.Sequential()

#input layer
model.add(tf.keras.layers.Flatten())
#hidden layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#output layer
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x, y, epochs=EPOCHS)
print("Created dataset successfully ")

#save model
model.save(model_name)
print("Saved model as " + model_name)
