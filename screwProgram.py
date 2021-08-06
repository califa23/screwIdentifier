import nueralnetwork as nn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#file = "G:\School\MachineLearning\myWorkAfterCollege\mnist\mnist.npz"


training_images = np.load("screws_images.npy")
training_labels = np.load("screws_labels.npy")


layer_sizes = (4096,30,2)
#print(training_images.shape)
print(training_labels.shape)


net = nn.NeuralNetwork(layer_sizes)
prediction = net.predict(training_images)

for i in prediction:
    print(i, np.argmax(i))
    print()

#plt.imshow(training_images[0].reshape(64,28),cmap=plt.cm.binary)
#plt.show()
