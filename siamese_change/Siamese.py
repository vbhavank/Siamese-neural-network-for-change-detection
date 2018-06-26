"""Designing the Siamese Twin sister network
   VGG16 is used with shared parameters"""
# coding: utf-8

# In[41]:


import keras
#import cv2


# In[42]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model

"""Function to take in imagenet weights for the Siamese Network"""
def siamese_network(input_image):
#    print('Inside Siamese')
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    #print model.summary()
    img_path = input_image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #print img
    #print x.shape
    feature = model.predict(x)
    #print feature.shape
    #print model.summary()
    return feature

#g = siamese_network('1.tiff')
#print g.shape
# In[43]:


#g = siamese_network('1.tiff')
#print g.shape


# In[33]:





# In[38]:





# In[ ]:




