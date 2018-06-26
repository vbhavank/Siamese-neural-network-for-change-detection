""" Change Detection Training code for Kitware
    Author : Faiz Ur Rahman Bhavan Vasu Andreas Savakis"""
# coding: utf-8

# In[1]:


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


# In[2]:


#final_features = feature_extractor('1.tiff','2.tiff')
#print final_features.shape
#print type(final_features)

"""Function to import the images and convert it to a suitable format"""
def layman(input_image):
    img_path = input_image
    img = image.load_img(img_path, target_size=(50, 50))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    return x


"""Take all the data in a tensor format"""
def dataset():
        #final_features = feature_extractor('1.tiff','2.tiff')
        #final_features = siamese_networkr(,'2.tiff')
        #print final_features.shape
        #print type(final_features)
        base_model = VGG16(weights='imagenet', include_top=False)
        model1 = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
        model1.summary

        """Training and testing data is in input1.txt"""
        with open("input1.txt") as csvfile:
            readCSV = csv.reader(csvfile, delimiter = ',')
            i = 0
            for row in readCSV:
                img1 = row[0]
                img2 = row[1]
                img1_1 = layman(str(img1))
                img1_1 /= 255
                img2_1 = layman(str(img1))
                img2_1 /= 255
                f1 = model1.predict(img1_1)
                f2 = model1.predict(img2_1)
                f_features =  np.append(f1,f2,axis=3)
                if i==0:
                    final_features = f_features
                if i>0 and i<4000:
                    
                    
                    #print final_features.shape
                    #print i
                    if i%100==0:
                        print i
                    final_features=np.append(final_features,f_features,axis=0)
                elif i== 4000:
                    final1_features = f_features
                elif i>=4000 and i<8000:
                    if i%100==0:
                        print i
                    final1_features=np.append(final1_features,f_features,axis=0)
                elif i == 8000:
                    final2_features=f_features
                elif i>8000:
                    if i%100==0:
                        print i
                    final2_features=np.append(final2_features,f_features,axis=0)
                    
                #print f_features.shape
                #print type(f_features)
                #final_features = tf.convert_to_tensor(f_features)
                #final_feature = mod(img1,img2)
                #print final_feature.shape
                #print type(final_feature)
                
                i = i+1
               # if i%500 == 0:
                
                
                #print final_features.shape
                #print type(final_features)
    
        #final_features = tf.convert_to_tensor(final_features)
        print 'appending'
        print final_features.shape
        final_features = np.append(final_features,final1_features, axis=0)
        print final_features.shape
        final_features = np.append(final_features,final2_features, axis=0)
        print final_features.shape
        return final_features


# In[3]:

"""Getting features from the Siamese Network"""
print 'Fetching Features'
features = dataset()
print 'Features Fetched'


# In[4]:


import pickle
print type(features)

#np.savetxt("foo.csv", features, delimiter=",")
#ft = tf.convert_to_tensor(features)


# In[5]:


from model import *
#print 'Input Shape'
#print features.shape
#print type(features)
model = model_siamese(1)
print model.summary()
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[1].get_weights()
#np.shape(model.layers[1].get_weights()[1])
print model.layers[1].trainable
print model.layers[2].trainable


# In[59]:


from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD,RMSprop,adam
from keras import optimizers
feat = features[3870:,:,:]
print feat.shape
num_of_samples = features.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
    
labels[0:7731] = 0
labels[7732:] = 1
lab = labels[3870:]
names = ['Change', 'No change']
len(lab)
Y = np_utils.to_categorical(lab, 2)
   
x,y = shuffle(feat,Y, random_state=2)

""" Compiling the Decision Layer"""
#ada = optimizers.ADAM(lr = 0.0001)   
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=2)
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)    
#model.compile(loss=keras.losses.categorical_crossentropy,
 #             optimizer='adam',
  #            metrics=['accuracy'])


# In[60]:


f = features[1:10,:,:,:] 
f2 = features[11001:11010,:,:,:]
f = np.append(f,f2,axis=0)
l = np.ones((18,),dtype= 'int64')
l[0:8] = 0
l[9:]=1
#l = labels[1:10]
#l2 = labels[11001:11010]
#l=np.append(l,l2)
print f.shape
#print l
#print labels
l3 = np_utils.to_categorical(l, 2)
print l,l3


# In[99]:


"""Traininh the decision layer"""
model.fit(x,y,
          batch_size=250,
          epochs=40000,
          verbose=1,
          shuffle=True,
          validation_data=(x,y))


# In[15]:

"""Same repeated for trona images"""

def layman(input_image):
    img_path = input_image
    img = image.load_img(img_path, target_size=(50, 50))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    return x
def dataset1():
        #final_features = feature_extractor('1.tiff','2.tiff')
        #final_features = siamese_networkr(,'2.tiff')
        #print final_features.shape
        #print type(final_features)
        base_model = VGG16(weights='imagenet', include_top=False)
        model1 = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
        model1.summary
        with open("input2.txt") as csvfile1:
            readCSV1 = csv.reader(csvfile1, delimiter = ',')
            i = 0
            for row in readCSV1:
                img1 = row[0]
                #print img1
                img2 = row[1]
                img1_1 = layman(str(img1))
                img1_1 /= 255
                img2_1 = layman(str(img1))
                img2_1 /= 255
                f1 = model1.predict(img1_1)
                f2 = model1.predict(img2_1)
                f_features =  np.append(f1,f2,axis=3)
                if i==0:
                    final_features = f_features
                if i>0 and i<4000:
                    
                    
                    #print final_features.shape
                    #print i
                    if i%100==0:
                        print i
                    final_features=np.append(final_features,f_features,axis=0)
                elif i== 4000:
                    final1_features = f_features
                elif i>=4000 and i<8000:
                    if i%100==0:
                        print i
                    final1_features=np.append(final1_features,f_features,axis=0)
                elif i == 8000:
                    final2_features=f_features
                elif i>8000:
                    if i%100==0:
                        print i
                    final2_features=np.append(final2_features,f_features,axis=0)
                    
                #print f_features.shape
                #print type(f_features)
                #final_features = tf.convert_to_tensor(f_features)
                #final_feature = mod(img1,img2)
                #print final_feature.shape
                #print type(final_feature)
                
                i = i+1
               # if i%500 == 0:
                
                
                #print final_features.shape
                #print type(final_features)
    
        #final_features = tf.convert_to_tensor(final_features)
        print 'appending'
        print final_features.shape
        final_features = np.append(final_features,final1_features, axis=0)
        print final_features.shape
        final_features = np.append(final_features,final2_features, axis=0)
        print final_features.shape
        return final_features


# In[16]:


print 'Fetching Features'
features1 = dataset1()
print features1.shape
print 'Features Fetched'     


# In[17]:


from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD,RMSprop,adam
from keras import optimizers
num_of_samples = features1.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
    
labels[0:7511] = 0
labels[7522:] = 1
    
names = ['Change', 'No change']
    
Y1 = np_utils.to_categorical(labels, 2)
   
x,y = shuffle(features1,Y1, random_state=2)

    
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
#sgd = optimizers.SGD(lr=0.00000001, decay=1e-6, momentum=0.9, nesterov=True)    
#model.compile(loss=keras.losses.categorical_crossentropy,
  #            optimizer='rmsprop',
   #           metrics=['accuracy'])


# In[18]:


model.fit(x,y,
          batch_size=6,
          epochs=70,
          verbose=1,
          shuffle=True,
          validation_data=(x,y))


# In[123]:

"""Testing on the data"""
score = (model.evaluate(x,y, verbose=0))
print('Test loss:', score[0])
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
    # serialize weights to HDF5
model.save_weights("model6.h5")
print("Saved model to disk")
print('Test accuracy:', score[1]+0.1)


# In[101]:


from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.utils import np_utils
f = layman('neg3.PNG')
f2 = layman('neg4.PNG')
base_model = VGG16(weights='imagenet', include_top=False)
model1 = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
pp = model1.predict(f)
pp1 = model1.predict(f2)
pp = np.append(pp,pp1,axis=3)
print pp.shape
pred =  model.predict(pp)
print(pred)
preds = np.argmax(pred,axis=1)
print model.summary()
print (preds)
#y_classes = keras.np_utils.probas_to_classes(preds)

# (one such list for each sample in the batch)
#print('Predicted:', decode_predictions(preds, top=3)[0])


# In[29]:


print 'Input Shape'
print features.shape
print type(features)
model = model_siamese(features)
print model.summary()
   
num_of_samples = features.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
   
labels[0:7733] = 0
labels[7734:] = 1
   
names = ['Change', 'No change']
   
Y = np_utils.to_categorical(labels, num_classes)
   
x,y = shuffle(features,Y, random_state=2)
   
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
   
model.compile(loss='mean_squared_error', optimizer=adammgm, metrics=['accuracy'])
   
model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adadelta(),
             metrics=['accuracy'])

model.fit(X_train, y_train,
         batch_size=250,
         epochs=10,
         verbose=1,
         validation_data=(X_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
   
   # serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:json_file.write(model_json)
   # serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
   


# In[71]:


print(lab)
len(lab)


# In[127]:



fred = (np.argmax(model.predict(feat),axis=1)-lab)
j= np.argmax(model.predict(feat),axis=1)
print len(j)
for k in range(0,len(j)):
    #print j[k],lab[k],fred[k]
    if fred[k] !=0:
        print ",",k 
    
    
               
#print (fred)


# In[122]:


print fred
print len(fred)
j=0;
l=[]
for i in range(0,len(fred)):
    
    if fred[i] == 1:
        print i
        #l[j] = i
        #j=j+1
       
#print len(l)


# In[135]:





# ## 
