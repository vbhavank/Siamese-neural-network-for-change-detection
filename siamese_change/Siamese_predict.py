"""Testing Code Enter the image path for testing
   Gives an output of class it belongs to"""
# coding: utf-8

# In[2]:


import keras as k
import cv2
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model
from Siamese import * 
from keras.layers import Flatten
from keras.layers import Dense
from Concatenate import *
from keras.models import Sequential
from keras.layers import Input
from model import *
import csv


# In[1]:


def layman(input_image):
    img_path = input_image
    img = image.load_img(img_path, target_size=(50, 50))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    return x



f1 = input('1st Image in quotes')
f3 = input('2nd Image in quotes')
f = layman(f1)

f /= 255


f2 = layman(f3)
f2 /= 255




"""Class 0 is change class 1 is no change"""
from keras.models import model_from_json
base_model = VGG16(weights='imagenet', include_top=False)
model1 = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
pp = model1.predict(f)
pp1 = model1.predict(f2)
pp = np.append(pp,pp1,axis=3)
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model6.h5")
print pp.shape
pred =  model.predict(pp)
preds = np.argmax(pred,axis=1)
print model.summary()
print (preds)
if preds==[0]:
    print('Change')
else:
    print 'No change'

