import cPickle as pickle
import os

pkl_root = './output/vgg/res/'
pkl_list = os.listdir(pkl_root)
pkl_list.sort()

for pkl in pkl_list:
    pkl_path = os.path.join(pkl_root, pkl)
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        print data
        break
