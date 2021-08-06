import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
EPOCHS = 30
imageres = 64

x = np.load("screws_images.npy")
y = np.load("screws_labels.npy")

print(x.shape)
print(y.shape)


##############

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

model.save('screws.model')

##############
imgfile = "screws/phillips1.jpg"
image = cv.imread(imgfile, cv.IMREAD_GRAYSCALE)
image = 255-image
width = imageres
height = imageres
dim = (width,height)
image = cv.resize(image,dim)
image = image / 255

pred = model.predict(image.reshape(1,imageres,imageres,1), batch_size=1)
if pred.argmax() == 1:
    print("Phillips")
else:
    print("Flathead")


plt.imshow(image,cmap=plt.cm.binary)
plt.show()
