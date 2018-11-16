# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 21:12:58 2018

@author: CCRG
"""

import tensorflow as tf


def model():
    
    
    
    xs = tf.placeholder(tf.float32, [None,32,32,3])/255.  # 32*32*3
    ys = tf.placeholder(tf.float32, [None, 10])
    
    keep_prob = tf.placeholder(tf.float32)
    
    global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    
    if keep_prob==1:
        drop_rate = [1,1,1]
    else:
        drop_rate = [0.25,0.5,0.5]
    
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
    
    conv1_1 = tf.layers.conv2d(
      inputs=xs,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'conv1_1'
      )
    layer_norm1_1 = tf.contrib.layers.layer_norm(conv1_1)
    
    conv1_2 = tf.layers.conv2d(
      inputs=layer_norm1_1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'conv1_2'
      )
    layer_norm1_2 = tf.contrib.layers.layer_norm(conv1_2)
    
    pool1 = tf.layers.max_pooling2d(layer_norm1_2 , pool_size=[2, 2], strides=2)
    
    
    dropout1 = tf.layers.dropout(
      inputs=pool1, rate=drop_rate[0])
    
    
    conv2_1 = tf.layers.conv2d(
      inputs=dropout1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'conv2_1')
    layer_norm2_1 = tf.contrib.layers.layer_norm(conv2_1)
    pool2_1 = tf.layers.max_pooling2d(layer_norm2_1  , pool_size=[2, 2], strides=2)
    
    conv2_2 = tf.layers.conv2d(
      inputs=pool2_1,
      filters=128,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'conv2_2')
    layer_norm2_2 = tf.contrib.layers.layer_norm(conv2_2)
    pool2_2 = tf.layers.max_pooling2d(layer_norm2_2 , pool_size=[2, 2], strides=2)
    
    
    dropout2 = tf.layers.dropout(
      inputs=pool2_2, rate=drop_rate[1])
    
    flat = tf.reshape(dropout2, [-1, 4 * 4 * 128])
      
    #pool3_flat = tf.reshape(pool3, [-1, 102400])
    
    dense1 = tf.layers.dense(flat, units=1500, activation=tf.nn.relu,kernel_regularizer=regularizer,name = 'dense1')
    layer_norm1 = tf.contrib.layers.layer_norm(dense1)
    dropout3 = tf.layers.dropout(
      inputs=layer_norm1, rate=drop_rate[2])
    
    logits = tf.layers.dense(inputs=dropout3, units=10,kernel_regularizer=regularizer,name = 'logits')
    
    prediction = tf.nn.softmax(logits)
    
    #cross_entropy = -tf.reduce_mean(ys*tf.log(prediction))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=ys))
    
    return prediction, loss, learning_rate

    
    
    
def lr(epoch):
    learning_rate = 1e-3
    if epoch > 80:
        learning_rate *= 0.5e-3
    elif epoch > 60:
        learning_rate *= 1e-3
    elif epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate