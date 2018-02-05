import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d as xavier_conv2d
from tensorflow.contrib.layers import xavier_initializer as xavier


def batch_normalize(name, tensor, phase_train):
  with tf.variable_scope(name):
    return tf.layers.batch_normalization(inputs=tensor, training=phase_train)

def conv(name, bottom, shape,
        padding='SAME', stride=[1,1,1,1], activation='Relu',
        batch_norm=False,phase_train=True):

  with tf.variable_scope(name):
    kernal = tf.get_variable(name='weights',shape=shape, initializer=xavier_conv2d())
    conv = tf.nn.conv2d(bottom, kernal, stride, padding=padding)
    bias = tf.get_variable(name='biases', shape=[shape[-1]], initializer=xavier())
    preactivation = tf.nn.bias_add(conv, bias)
    if batch_norm:
      preactivation = batch_normalize(name='bn',tensor=preactivation,phase_train=phase_train)
    if activation=='Relu':
      relu=tf.nn.relu(preactivation, name=name)
      return relu
    return preactivation

def rconv(name, bottom, shape, iteration):
  with tf.variable_scope(name):
    ckernal = tf.get_variable(name='forward weights', shape=shape, initializer=xavier_conv2d())
    rkernal = tf.get_variable(name='recurrent weigths', shape=shape, initializer=xavier_conv2d())
    fbias = tf.get_variable(name='forward bias', shape=[shape[-1]], initializer=xavier())
    rbias = tf.get_variable(name='recurrent bias',shape=[shape[-1]], initializer=xavier())
    
    rconv = tf.nn.conv2d(bottom, fkernal, shape, 'SAME')
    rconv = tf.nn.bias_add(conv, fbias)

    for i in range(iteration-1):
      conv = tf.nn.conv2d(bottom, fkernal, shape, 'SAME')
      conv = tf.nn.bias_add(conv, fbias)

      rconv = tf.nn.conv2d(rconv, rkernal, shape, 'SAME')
      rconv = tf.nn.bias_add(rconv, rbias)
      
      rconv = conv + rconv

    return rconv

def fc(name, bottom, top, activation='Relu'):
  with tf.variable_scope(name):
    shape = bottom.get_shape().as_list()
    dim = np.prod(shape[1:])
    reshape = tf.reshape(bottom,[-1,dim])
    weights = tf.get_variable(name='weights',shape=[dim,top], initializer=xavier(uniform=False))
    biases = tf.get_variable(name='biases', shape=[top], initializer=xavier())
    preactivation = tf.matmul(reshape, weights) + biases
    if activation == 'Relu':
      relu = tf.nn.relu(preactivation, name=name)
      return relu
    return preactivation

def pool(name, bottom, ksize=[1,2,2,1], strides = [1,2,2,1]):
  return tf.nn.max_pool(bottom, ksize=ksize, strides=strides, padding='SAME', name=name)
