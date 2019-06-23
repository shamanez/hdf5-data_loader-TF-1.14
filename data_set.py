import h5py
import tensorflow as tf

class generator:
    def __init__(self, hdf5_file, d_sam='sample',l_sam='label'):
        self.file = hdf5_file
        self.samples=d_sam  #The dictionary key of your hdf5 file which contains the training data 
        self.labels=l_sam  #The dictionary key of your hdf5 file which contain the labels of training data (Note : indexing should be same)

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for im,label in zip(hf[self.samples],hf[self.labels]):
                yield im,label