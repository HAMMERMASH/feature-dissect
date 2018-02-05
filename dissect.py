import os

import tensorflow as tf
import numpy as np

from vgg16 import vgg16

import scipy.misc as misc
import time

import _pickle as pickle

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

VID_DIR = '/slwork/VID/ILSVRC/Data/VID/val/'
CHECKPOINT_PATH = 'vgg_16.ckpt'
SAVE_DIR = 'output/vgg/'


def res_initial(channels, nums):
  res = []
  for channel, num in zip(channels, nums):
    for i in range(num): 
      res.append(np.zeros((channel),np.float32))
  return res
if __name__ == '__main__':
  with tf.Graph().as_default():
    net = vgg16() 
    print('Build graph...')
    net.build()
    sess = tf.Session()
    print('Restore from {}...'.format(CHECKPOINT_PATH))
    net.restore(sess, CHECKPOINT_PATH)


    video_list = os.listdir(VID_DIR)
    video_list.sort()
    
    num_video = len(video_list)
    count = 0
    for video in video_list:
      if '2017_' in video:
        print('Processing {}...'.format(video))

        image_list = os.listdir(VID_DIR+video)
        image_list.sort()
        old_blobs = []

        res = res_initial([64,128,256,512,512],[2,2,3,3,3])
        time_start = time.clock()
        for image_name in image_list:
          if 'JPEG' in image_name:
            image_path = VID_DIR+video+'/'+image_name
            image = misc.imread(image_path)
            time_start = time.clock() 
            blob_list = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3']
            save_list = []
            blobs = net.get_blob(sess, blob_list, image)
            if len(old_blobs) > 0:
              ind = 0
              for new_blob, old_blob in zip(blobs, old_blobs):
                num_pixel = np.prod(new_blob.shape[1:])
                residual = new_blob-old_blob
                shape = residual.shape
                residual = np.abs(np.sum(np.reshape(residual,[-1,shape[-1]]),axis=0)) / num_pixel
                res[ind] += residual
                ind += 1

            for item in save_list:
              feature_map = blobs[item[0]][0][:,:,item[1]]
              zeros = np.zeros(feature_map.shape)
              feature_map = np.stack((feature_map,zeros,zeros),axis=2)
              misc.imsave(SAVE_DIR+image_name.split('.')[0]+'_'+blob_list[item[0]]+'_{}'.format(item[1])+'.jpg',feature_map)
                
            old_blobs = blobs
            print(image_name)
        print(time.clock() - time_start) 
        count += 1
        print('{}/{}'.format(count, num_video))
        data = pickle.dumps(res)
        pickle.dump(data, open('{}.pkl'.format('output/vgg/res/{}'.format(video)), 'wb'))
        print('Result saved in {}.pkl'.format(video))

