# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:41:29 2018

@author: CCRG
"""

import tensorflow as tf
import os


weights = tf.get_default_graph().get_tensor_by_name(
  os.path.split(conv1_1.name)[0] + '/kernel:0')

bias = tf.get_default_graph().get_tensor_by_name( os.path.split(conv1_1.name)[0] + '/bias:0')

weights1 = weights.eval(session=sess)

bias1 = bias.eval(session = sess)
########################################################################
weights = tf.get_default_graph().get_tensor_by_name(
  os.path.split(conv1_2.name)[0] + '/kernel:0')

bias = tf.get_default_graph().get_tensor_by_name( os.path.split(conv1_2.name)[0] + '/bias:0')

weights2 = weights.eval(session=sess)

bias2 = bias.eval(session = sess)
##############################################################################

weights = tf.get_default_graph().get_tensor_by_name(
  os.path.split(conv2_1.name)[0] + '/kernel:0')

bias = tf.get_default_graph().get_tensor_by_name( os.path.split(conv2_1.name)[0] + '/bias:0')

weights3 = weights.eval(session=sess)

bias3 = bias.eval(session = sess)
#################################################################

weights = tf.get_default_graph().get_tensor_by_name(
  os.path.split(conv2_2.name)[0] + '/kernel:0')

bias = tf.get_default_graph().get_tensor_by_name( os.path.split(conv2_2.name)[0] + '/bias:0')

weights4 = weights.eval(session=sess)

bias4 = bias.eval(session = sess)







