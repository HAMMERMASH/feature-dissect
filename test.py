import _pickle as pickle
import numpy as np

data = pickle.load(open('ILSVRC2017_val_00000000.pkl', 'rb'))
info = pickle.loads(data)

print(info)
