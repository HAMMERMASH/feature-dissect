import tensorflow as tf
import numpy as np
from layer import conv, fc, pool

class vgg16:
  def __init__(self):
    self._feature_map = {} 
    self.input = tf.placeholder(tf.float32, [None,None,None,3]) 

  def preprocess(self, images):
    r, g, b = tf.split(axis=3, num_or_size_splits=3, value=images)
    return tf.concat([r-123.68,g-116.78,b-103.94], axis=3)

  def build_conv(self, name, bottom, shape, repeat):
    with tf.variable_scope(name):
      net = conv(name+'_1', bottom, [3,3,shape[0],shape[1]])
      self._feature_map[name+'_1'] = net
      for i in range(repeat-1):
        net = conv(name+'_{}'.format(i+2), net, [3,3,shape[-1],shape[-1]])
        self._feature_map[name+'_{}'.format(i+2)] = net
      net = pool('pool'+name[-1], net)
      self._feature_map['pool'+name[-1]] = net
      return net

  def build(self):
    with tf.variable_scope('vgg_16'):
      p_images = self.preprocess(self.input)
      net = self.build_conv('conv1',p_images , [3,64], 2)
      net = self.build_conv('conv2', net, [64,128], 2)
      net = self.build_conv('conv3', net, [128,256], 3)
      net = self.build_conv('conv4', net, [256,512], 3)
      net = self.build_conv('conv5', net, [512,512], 3)
      
    self.restorer = tf.train.Saver()

  def restore(self, sess, path):
    self.restorer.restore(sess, path)
      
  def get_blob(self, sess, blob_list, image):
    blobs = [self._feature_map[t] for t in blob_list] 
    if len(image.shape) == 3:
      image = np.expand_dims(image, axis=0)
    feed_dict={self.input: image}
    blob_array = sess.run(blobs, feed_dict=feed_dict)
    return blob_array
