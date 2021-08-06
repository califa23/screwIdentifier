#John Califano
#7/24/2021
#This file creates a dataset tensorflow training
#adjust the image count(always even) and the imagres
#creates two file at end, screw_images, and screw labels


#TODO: make this executable through cli or cmd or a class
#      make create one output file


import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

print("Creating datset...")
outfilename = "screws"
filepath = ".\screws\\"
imagecount = 18
imageres = 64
images = np.empty([imagecount,imageres,imageres])
labels = np.empty([imagecount,])

for i in range(int(imagecount/2)):
    file = filepath + "phillips"+str(i+1)+".jpg"
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    image = 255-image
    width = imageres
    height = imageres
    dim = (width,height)
    image = cv.resize(image,dim)
    image = image / 255
    images[i] = image
    labels[i] = 1.0
    
for i in range(int(imagecount/2)):
    file = filepath + "other"+str(i+1)+".jpg"
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    image = 255-image  
    width = imageres
    height = imageres
    dim = (width,height)
    image = cv.resize(image,dim)
    image = image / 255
    images[i+int((imagecount/2))] = image
    labels[i+int((imagecount/2))] = 0.0

print('images shape: {0}'.format(images.shape))
print('labels shape: {0}'.format(labels.shape))



np.save("screws_images",images)
np.save("screws_labels",labels)

print("Saved two files: screws_images.npy, screws_labels.npy")


