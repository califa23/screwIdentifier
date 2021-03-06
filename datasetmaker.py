import os
import numpy as np
import cv2 as cv

filepath = ".\images\\"
label = ["phillips","slotted"]
imagecount = 18
imageres = 64
images = np.empty([imagecount,imageres,imageres])
labels = np.empty([imagecount,])

index = 0
for i in range(len(label)):
    for file in os.listdir(filepath):
        if label[i] in file:
            print("Processing "+file)
            image = cv.imread(filepath+file, cv.IMREAD_GRAYSCALE)
            image = 255-image
            width = imageres
            height = imageres
            dim = (width,height)
            image = cv.resize(image,dim)
            image = image / 255
            images[index] = image
            labels[index] = i
            index += 1


print('images shape: {0}'.format(images.shape))
print('labels shape: {0}'.format(labels.shape))

np.save("images",images)
np.save("labels",labels)

print("Saved two files: images.npy, labels.npy")



