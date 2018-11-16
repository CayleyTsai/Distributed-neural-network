# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:13:31 2018

@author: CCRG
"""

import sys
from sklearn.model_selection import KFold
import numpy as np
from keras.datasets import cifar10

from keras.utils import np_utils

from keras.layers import Conv2D,Dense, Activation, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

import sys
import matplotlib.pyplot as plt

x_train_ = np.load('40000_train_x.npy')
y_train_ = np.load('40000_train_y.npy')
x_test = np.load('test_x.npy')
y_test = np.load('test_y.npy')
#data preprocess

num_classes = 10
import keras

y_train_ = keras.utils.to_categorical(y_train_, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train_ = x_train_.astype('float32')
x_test = x_test.astype('float32')

# data preprocessing
x_train_[:,:,:,0] = (x_train_[:,:,:,0]-123.680)
x_train_[:,:,:,1] = (x_train_[:,:,:,1]-116.779)
x_train_[:,:,:,2] = (x_train_[:,:,:,2]-103.939)
x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)

x_train_ = x_train_/255.
x_test = x_test/255.


# fine tune drop out
hc_list = []
model_list = []
drop_rate = [0.5]

for drop1 in drop_rate:
    for drop2 in drop_rate:
        #cross validation
        validation_count = 0
        kf = KFold(n_splits = 5)
        
        
        for train_index, val_index in kf.split(x_train_):
            validation_count += 1
            X_train = x_train_[train_index] 
            X_val = x_train_[val_index]
            Y_train = y_train_[train_index]
            Y_val = y_train_[val_index]
            model = Sequential()
            
            layer_name = ['pull_out1','pull_out2','pull_out3']
            
            model.add(Conv2D(batch_input_shape=(None,32,32,3),
                             filters = 32,strides=1,
                             kernel_size=3,
                             padding='same',

                             ))
            model.add(Conv2D(batch_input_shape=(None,32,32,32),
                             filters = 32,strides=1,
                             kernel_size=3,
                             padding='same',

                             ))
            
            model.add(Activation('relu'))
            model.add(MaxPooling2D(
                    pool_size=2,
                    strides=2,
                    padding='same',

                    name = 'pull_out1'
                    ))
            
            
            
            model.add(Conv2D(batch_input_shape=(32,16,16),
                             filters = 64,strides=1,
                             kernel_size=3,
                             padding='same',

                             ))
            model.add(Conv2D(batch_input_shape=(64,16,16),
                             filters = 64,strides=1,
                             kernel_size=3,
                             padding='same',

                             ))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(
                    pool_size=2,
                    strides=2,
                    padding='same',
                    data_format = 'channels_first',
                    name = 'pull_out2'
                    ))
            model.add(Dropout(drop1))
            
        
            
            print('intermediate model okay')
            
            
            
            model.add(Flatten())
            
            model.add(Dense(1000))
            model.add(Activation('relu'))
            model.add(Dropout(drop2))
            model.add(Dense(10))
            model.add(Activation('softmax'))
            
            from keras.models import Model
            
            intermediate_layer_model1 = Model(inputs = model.input,outputs=model.get_layer(layer_name[0]).output)
            intermediate_layer_model2 = Model(inputs = model.input,outputs=model.get_layer(layer_name[1]).output)
            
            #optimize
            adam = Adam(lr=1e-4)
            
            model.compile(optimizer=adam,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            
            
            print('droprate = ',drop2)
            print('val = ',validation_count)
            print('Training ------------ ')
            
            history_center = model.fit(x = X_train, y = Y_train, validation_data=(X_val,Y_val), epochs = 200, batch_size=64)
            
            
            print('\nTesting ------------')
            # Evaluate the model with the metrics we defined earlier
            loss, accuracy = model.evaluate(x_test, y_test)
            
            print('\ntest loss: ', loss)
            print('\ntest accuracy: ', accuracy)
            
            model_list.append(model)
            hc_list.append(history_center)
            
            
            break;
            
            #respectively train local model
            
            
            #hidden layer 1
            # get hidden output
            
            
            #hidden layer2
            #get hidden output
            
            
            
            
            #plot result
            
           
            
            
            # summarize history for accuracy
#            print('drop out = ',drop)
#            plt.plot(history_center.history['acc'])
#            plt.plot(history_center.history['val_acc'])
#            plt.title('model accuracy')
#            plt.ylabel('accuracy')
#            plt.xlabel('epoch')
#            plt.legend(['train', 'val'], loc='upper left')
#            plt.show()
#            # summarize history for loss
#            plt.plot(history_center.history['loss'])
#            plt.plot(history_center.history['val_loss'])
#            plt.title('model loss')
#            plt.ylabel('loss')
#            plt.xlabel('epoch')
#            plt.legend(['train', 'val'], loc='upper left')
#            plt.show()
            
            