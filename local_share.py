# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 05:20:28 2018

@author: CCRG
"""
import numpy as np
import tensorflow as tf

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre1 = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre1,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def compute_branch_accuracy(v_xs, v_ys):
    global prediction2
    y_pre2 = sess.run(prediction2, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre2,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#share_out_all_SmallStruct.npy

#build local network

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None,16,16,32])/255.  # 32*32*3
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#3X3X32

W_conv1 = weight_variable([3,3, 32,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32


#dense50

W_fc1 = weight_variable([8*8*32, 50])
b_fc1 = bias_variable([50])

h_pool_flat = tf.reshape(h_pool1, [-1, 8*8*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#dense10(output)
W_fc2 = weight_variable([50, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = -tf.reduce_mean(ys*tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
#Loss2


train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)


sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
    print('init1')
else:
    init = tf.global_variables_initializer()
    print('init2')
sess.run(init)

#read data
x_train_ = np.load('share_out_all_SmallStruct.npy')
y_train_ = np.load('10000_train_y.npy')
import keras
num_classes = 10

y_train_ = keras.utils.to_categorical(y_train_, num_classes)

x_train = x_train_[0:8000]
x_val = x_train_[8000:10000]
y_train = y_train_[0:8000]
y_val = y_train_[8000:10000]

EPOCH = 0
BATCH_INDEX = 0
BATCH_SIZE = 128
TRAIN_SIZE = 8000

local_val_loss = []
local_val_acc = []
local_acc_list = []
local_loss_list = []

while EPOCH < 201:

    batch_ys = y_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE,:]
    batch_xs = x_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE,:,:]
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
     
    BATCH_INDEX += BATCH_SIZE
    if BATCH_INDEX >= TRAIN_SIZE:
        print('epoch ',EPOCH)
        acc = compute_accuracy(
            x_train, y_train)
        print('accuracy : ',acc)
        local_acc_list.append(acc)
        loss = sess.run(cross_entropy,feed_dict = {xs:x_train, ys: y_train,keep_prob: 1})
        print('loss : ',loss)
        local_loss_list.append(loss)
        acc = compute_accuracy(
            x_val, y_val)
        print('val accuracy : ',acc)
        local_val_acc.append(acc)
        loss = sess.run(cross_entropy,feed_dict = {xs:x_val, ys: y_val,keep_prob: 1})
        print('val loss : ',loss)
        print('\n')
        local_val_loss.append(loss)
        EPOCH += 1
        BATCH_INDEX = 0


