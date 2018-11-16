# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:23:45 2018

@author: CCRG
"""


import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.initializers import he_normal
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
import sys

TRAINABLE = False

#data process
x_train_ = np.load('10000_train_x.npy')
y_train_ = np.load('10000_train_y.npy')

x_test = np.load('test_x.npy')
y_test = np.load('test_y.npy')

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



x_train = x_train_[0:8000]
x_val = x_train_[8000:10000]
y_train = y_train_[0:8000]
y_val = y_train_[8000:10000]


#build model
model = Sequential()
# Block 1
model.add(Conv2D(32, (3, 3), padding='same',name='block1_conv1', input_shape=x_train.shape[1:],trainable = TRAINABLE))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv2',trainable = TRAINABLE))
model.add(Activation('relu'))
#model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

## Block 2
#model.add(Conv2D(64, (3, 3), padding='same',name='block2_conv1',trainable = TRAINABLE))
#model.add(Activation('relu'))
#model.add(Conv2D(128, (3, 3), padding='same',name='block2_conv2',trainable = TRAINABLE))
#model.add(Activation('relu'))
#model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))


##local nn
#model.add(Conv2D(64,(3,3),padding='same',name='local_conv1'))
#model.add(Activation('relu'))
#model.add(Conv2D(128,(2,2),padding='same',name='local_conv2'))
#model.add(Activation('relu'))
model.add(Flatten(name='flatten'))

model.add(Dense(10))
model.add(Activation('softmax'))

adam = optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    
model.layers[0].set_weights([weights1,bias1])
    
model.layers[2].set_weights([weights2,bias2])

model.layers[5].set_weights([branch_weights,branch_bias])

logit = model.layers[6].output[:100]

model.predict(x_train[0:5])

    
#    model.layers[5].set_weights([weights3,bias3])
#    
#    model.layers[7].set_weights([weights4,bias4])

#history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=100, verbose=1,validation_data= (x_val,y_val))
print(model.evaluate(x_test,y_test))


