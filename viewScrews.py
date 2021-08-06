import numpy as np
import matplotlib.pyplot as plt



images = np.load("screws_images.npy")
labels = np.load("screws_labels.npy")
print(len(images))


for i in range(len(images)):
    plt.imshow(images[i].reshape(64,64),cmap=plt.cm.binary)
    print(images[i]," ",labels[i])
    plt.show()
