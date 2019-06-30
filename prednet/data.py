from keras.utils import Sequence, to_categorical
from keras.preprocessing import image
from keras import backend as K

import glob
import os
import numpy as np
import pickle as pkl
import random as rn


# Getting reproducible results:
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)


class DataGenerator(Sequence):
    """
    Generates data for Keras.
    Adapted from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    """
    def __init__(self, batch_size=16, shuffle=False, fn_preprocess=None, seq_length=None, sample_step=1,
                 target_size=None, classes=None, data_format=K.image_data_format(), output_mode=None,
                 rescale=None, return_sources=False):
        
        self.batch_size = batch_size
        self.X = []
        self.y = []
        # self.sources = []
        self.data_dir = None
        self.shuffle = shuffle
        self.seq_length = seq_length
        self.sample_step = sample_step
        self.target_size = target_size
        self.fn_preprocess = fn_preprocess
        self.classes = classes
        self.data_format = data_format
        self.output_mode = output_mode
        self.rescale = rescale
        self.return_sources = return_sources
        
    def flow_from_directory(self, data_dir):
        self.data_dir = data_dir
        
        if self.classes is None:
            self.classes = sorted(next(os.walk(data_dir))[1])
        data_pattern = '{}/*'
        
        total_samples = 0
        for i, c in enumerate(self.classes):
            class_samples = sorted(glob.glob(os.path.join(data_dir, data_pattern.format(c))))
            self.__process_class_samples(i, class_samples)
            total_samples += len(class_samples)

        msg = 'Found {} samples belonging to {} classes in {}'
        print(msg.format(total_samples, len(self.classes), self.data_dir))
        self.__postprocess()
        return self
        
    def __process_class_samples(self, class_index, class_samples):
        self.y.extend([class_index] * len(class_samples))
        if self.seq_length:
            for i, sample in enumerate(class_samples):
                self.X.append([sample] * self.seq_length)
        else:
            self.X.extend(class_samples)

    def __postprocess(self):
        self.data_shape = self.__load_data(0).shape
        self.n_classes = len(self.classes)
        self.on_epoch_end()
        
    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """
        Generates data containing batch_size samples
        """
        # Initialization
        X = np.empty((self.batch_size,) + self.data_shape)
        y = np.empty(self.batch_size, dtype=int)
        sources = []

        # Generate data
        for i, index in enumerate(indexes):
            X[i] = self.__load_data(index)
            
            # Store class
            y[i] = self.y[index]
            sources.append(self.X[index])

        if self.data_format == 'channels_first':
            X = np.transpose(X, (0, 1, 4, 2, 3))
         
        if self.output_mode is not None and self.output_mode == 'error':  
            data = (X, np.zeros(self.batch_size, np.float32))
        else:
            data = (X, to_categorical(y, num_classes=self.n_classes))

        if self.return_sources:
            data += (np.array(sources),)
        return data
    
    def __preprocess(self, img):
        if self.rescale:
            img = self.rescale * img
        if self.fn_preprocess:
            img = self.fn_preprocess(img)
        return img
    
    def __load_image(self, filename):
        img = image.load_img(filename, target_size=self.target_size)
        img = image.img_to_array(img)
        # img = imread(filename)
        return self.__preprocess(img)
    
    def __load_pickle(self, filename):
        with open(filename, 'rb') as f:
            return pkl.load(f)
        
    def __load_sample(self, filename):
        if filename.lower().endswith('.pkl'):
            sample = self.__load_pickle(filename)
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            sample = self.__load_image(filename)
        elif filename == 'padding':
            sample = np.zeros(self.sample_shape)
        else:
            raise ValueError('{} format is not supported'.format(filename))
            
        self.sample_shape = sample.shape
        return sample
    
    def __load_seq_data(self, index):
        seq = self.X[index]
        seq_data = []
        
        for sample in seq:
            seq_data.append(self.__load_sample(sample))
        
        return np.array(seq_data)
    
    def __load_data(self, index):
        if len(self.X) <= index:
            return None
        
        if self.seq_length:
            data = self.__load_seq_data(index)
        else:
            data = self.__load_sample(self.X[index])
        return data
