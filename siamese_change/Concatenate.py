"""This code is to take the input from the siamese network 
   Features from each sister of Siamese Network to concatenate and form 
   big feature map"""
# coding: utf-8

# In[29]:


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


# In[32]:


def concatenate_features(f1,f2,axi_s):
    f_features = np.append(f1,f2,axis=axi_s)
    return f_features

    
    
    
def feature_extractor(img1,img2):
    feature_1=siamese_network(img1)
    feature_2=siamese_network(img2)
    
    
    
    final_features = concatenate_features(feature_1,feature_2,3)
    
    return final_features

