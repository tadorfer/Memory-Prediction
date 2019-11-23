import h5py
import numpy as np

path = '/Users/thomasdorfer/Desktop/Projects/MemoryPrediction/Data/'
# arrays = {}
# f = h5py.File(path+'thomasdata.mat')
# for k, v in f.items():
#     arrays[k] = np.array(v)


with h5py.File(path+'thomasdata.mat', 'r') as file:
    print(list(file.keys()))

# with h5py.File(path+'thomasdata.mat', 'r') as file:
#     a = list(file['a'])