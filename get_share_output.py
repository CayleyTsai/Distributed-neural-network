# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 05:20:28 2018

@author: CCRG
"""
import numpy as np

xx_train_ = np.load('10000_train_x.npy')
yy_train_ = np.load('10000_train_y.npy')

num_classes = 10
import keras

yy_train_ = keras.utils.to_categorical(yy_train_, num_classes)

xx_train_ = xx_train_.astype('float32')

# data preprocessing
xx_train_[:,:,:,0] = (xx_train_[:,:,:,0]-123.680)
xx_train_[:,:,:,1] = (xx_train_[:,:,:,1]-116.779)
xx_train_[:,:,:,2] = (xx_train_[:,:,:,2]-103.939)

#get share output
share_out = sess.run(h_pool1,feed_dict = {xs:xx_train_[:10000],ys:yy_train_[:10000],keep_prob: 1})

