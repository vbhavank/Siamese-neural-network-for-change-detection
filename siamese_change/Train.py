""" Change Detection Training code for Kitware
    Author : Faiz Ur Rahman Bhavan Vasu Andreas Savakis"""


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
from model import *
import csv
from keras.models import model_from_json




def dataset():

        with open("input1.txt") as csvfile:
            readCSV = csv.reader(csvfile, delimiter = ',')
            #i = 0
            for row in readCSV:
                img1 = row[0]
                img2 = row[1]
                
                final_feature = mod(img1,img2)
                final_features=np.append(final_features,final_feature,axis=0)
        final_features = tf.convert_to_tensor(final_features)
        return final_features

def dataset_testing():

        with open("input2.txt") as csvfile:
            readCSV = csv.reader(csvfile, delimiter = ',')
            #i = 0
            for row in readCSV:
                img1 = row[0]
                img2 = row[1]
                
                final_feature = model(img1,img2)
 
                final_features=np.append(final_features,final_feature,axis=0)

    
        final_features = tf.convert_to_tensor(final_features)
        return final_features

def training_part():
    print 'Fetching Features'
    features = dataset()
    print 'Features Fetched'
    model = model_siamese(features)
    print model.summary()
    
    num_of_samples = features.shape[0]
    labels = np.ones((num_of_samples,),dtype='int64')
    
    labels[0:15243] = 0
    labels[15244:] = 1
    
    names = ['Change', 'No change']
    
    Y = np_utils.to_categorical(labels, num_classes)
    
    x,y = shuffle(features,Y, random_state=2)
    
   
    
    model.compile(loss='mean_squared_error', optimizer=adammgm, metrics=['accuracy'])
    
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    model.fit(x, y,
          batch_size=250,
          epochs=epochs,
          verbose=1,
          validation_data=(X_train, y_train))
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
          json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model6.h5")
    print("Saved model to disk")
    
    #Testing
    features = dataset_testing()

    
    num_of_samples = features.shape[0]
    labels = np.ones((num_of_samples,),dtype='int64')
    
    labels[0:5000] = 0
    labels[5000:] = 1
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model6.h5")
    names = ['Change', 'No change']
    
    
    Y = np_utils.to_categorical(labels, num_classes)
    
    x_test,y_test = shuffle(features,Y, random_state=2)
    
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    
    
    
    
training_part()
