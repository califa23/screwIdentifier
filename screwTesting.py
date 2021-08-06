import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

print('Testing...')
imageres = 64

model=tf.keras.models.load_model('screws.model')

imgfile = "screwImages/test2.jpg"
print("Test data: " + imgfile)
image = cv.imread(imgfile, cv.IMREAD_GRAYSCALE)
image = 255-image
width = imageres
height = imageres
dim = (width,height)
image = cv.resize(image,dim)
image = image / 255

pred = model.predict(image.reshape(1,imageres,imageres,1), batch_size=1)

if pred.argmax() == 0:
    label = "Phillips"
else:
    label = "Slotted"

plt.title(label)
plt.imshow(image,cmap=plt.cm.binary)
plt.show()

