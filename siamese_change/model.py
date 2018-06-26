"""Architecture of final decision layer"""
# coding: utf-8

# In[4]:


import keras as k
#import cv2
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


# In[13]:

"""Calling the convolution part of the network"""
def mod(img1,img2):
    final_features = feature_extractor(str(img1),str(img2))
    return final_features
    
"""Decision layer containing 3 Fully Connected layers
   With 2 class outputs change or no change"""
def model_siamese(features):
    
    #final_features=feature_extractor('1.tiff','2.tiff')
    #print type(final_features)
    
    #model
   # y = preprocess_input(final_features)
    
    """model = Sequential()
    model.add(Flatten(),input_shape=(1,7,7,1024))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(2, activation='softmax'))"""
    x1 = Input((1,1,1024))
    x = Flatten(name='flatten')(x1)
    #x = Flatten(name='flatten')(x)
    x = Dense(4096,activation='softmax', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(2, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=x1, outputs=x)
    return model    
    #print model.summary()
    #print model.predict(features)
    
    


# In[14]:


#model_siamese()

