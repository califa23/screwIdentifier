#John Califano
#7/24/2021
#This file creates a dataset tensorflow training
#adjust the image count(always even) and the imagres
#creates two files at end, screw_images, and screw labels


#TODO: make this executable through cli or cmd or a class
#      make create one output file (maybe not)


import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


outfilename1 = "screws"
outfilename2 = "screws"
filepath = "G:\School\MachineLearning\myWorkAfterCollege\screws\\"
imagecount = 18
imageres = 64
images = np.empty([imagecount,imageres,imageres])
labels = np.empty([imagecount,])

def create():
    print("Creating datset...")
    for i in range(int(imagecount/2)):
        file = filepath + "phillips"+str(i+1)+".jpg"
        image = cv.imread(file, cv.IMREAD_GRAYSCALE)#loadfile in b+w
        image = 255-image#inverts image cuz its loaded other way
        width = imageres #set desired width of img
        height = imageres#set desired height of img
        dim = (width,height)#create dimension tuple from two values above
        image = cv.resize(image,dim)#resize loaded image to dimension above
        image = image / 255 #normalize values
        images[i] = image #load 2d image into array 3d array
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

def save():
    try:
        np.save("screws_images",images)
        np.save("screws_labels",labels)
        print("SUCCESS: Saved two files: {0}, {1}".format(outfilename1,outfilename2))
    except:
        print("ERROR: Failed to save file")


