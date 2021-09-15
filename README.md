# Screw Identifier
## Overview
Using openCV and tensorflow we can identify different types of screw heads. This basic demonstration can be expanded on and can have many applications in the real world.
Some of these applications include construction, manufacturing, and machine repair work in environments that may be too hostile for humans. 

## Files

* datasetmaker.py
  * This file taks the images from the image folder and creates a dataset of necessary dimensions to train the network.
* modelmaker.py
  * This file uses tensorflow to train and create a model from the dataset provided by datasetmaker.py.
* tester.py
  * This is the file where we can test our model with new images.
* viewing.py
  * This file allows us to view the images in the created dataset from datasetmaker.py. (Mainly for debugging purposes.)

## More Information
For more information and a detailed walk through: http://johncalifano.info/#/app-projects-screw-identifier
