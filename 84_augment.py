# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 03:36:36 2018

@author: CCRG
"""

from __future__ import print_function
import tensorflow as tf
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre1 = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre1,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def compute_all_accuracy(v_xs, v_ys):
    global prediction
    sum_correct = 0
    data_size = len(v_xs)
    for i in range(data_size):  
       y_pre1 = sess.run(prediction, feed_dict={xs: v_xs[np.newaxis,i], keep_prob: 1})
       if np.argmax(y_pre1,1) == np.argmax(v_ys[i]):
           sum_correct += 1
      
   
    return sum_correct / data_size
    
    

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None,32,32,3])/255.  # 32*32*3
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

if keep_prob==1:
    drop_rate = [1,1,1]
else:
    drop_rate = [0.25,0.5,0.5]

regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

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

#not used
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


global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)



optimizer = tf.train.AdamOptimizer(0.001)

grads = optimizer.compute_gradients(loss)
for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 5), v)
train_op = optimizer.apply_gradients(grads)


#train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy1)

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

#data process
x_train_ = np.load('40000_train_x.npy')
y_train_ = np.load('40000_train_y.npy')

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



x_train = x_train_[0:32000]
x_val = x_train_[32000:40000]
y_train = y_train_[0:32000]
y_val = y_train_[32000:40000]


print('Using real-time data augmentation.')
datagen = ImageDataGenerator(horizontal_flip=True,
        width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

datagen.fit(x_train)

batch_size   = 128
epochs       = 100
iterations   = 391



#EPOCH = 1
#BATCH_INDEX = 0
#BATCH_SIZE = 50
#TRAIN_SIZE = 32000

branch_loss = []
branch_acc = []
acc_list = []
loss_list = []
val_acc_list = []
val_loss_list = []


for i in range(epochs):
    for j in range(iterations):
        tup = datagen.flow(x_train,y_train,batch_size=batch_size)[0]
        batch_xs = tup[0]
        batch_ys = tup[1]
        sess.run(train_op, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    
    print('epoch ',i)
    #main
    print('*main')
    acc = compute_accuracy(
        x_train[0:1000,:], y_train[0:1000,:])
    print('train accuracy : ',acc)


    acc_list.append(acc)
    #debugging softmax output
    #score = sess.run(prediction1,feed_dict = {xs:x_train[0:1,:,:], ys: y_train[0:1,:],keep_prob: 1})
    #print(score)
    loss1 = sess.run(loss,feed_dict = {xs:x_train[0:1000,:,:], ys: y_train[0:1000,:],keep_prob: 1})
    print('loss : ',loss1)
    loss_list.append(loss1)
    #main val
    val_acc = compute_all_accuracy(
            x_val,y_val
            )
    print('val accuracy',val_acc)
    loss2 = sess.run(loss,feed_dict = {xs:x_val[0:1000,:,:], ys: y_val[0:1000,:],keep_prob: 1})
    print('val loss',loss2)
    val_acc_list.append(val_acc)
    val_loss_list.append(loss2)







