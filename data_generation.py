import numpy as np
import keras
import matplotlib.image as mpimg

from image_preprocessing import preprocess_image

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, labels, batch_size=32, dim=(160, 320), n_channels=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples""" # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=np.float32)

        # Generate data
        for i, curr_image_path in enumerate(list_IDs_temp):
            # Store sample
            curr_sample = mpimg.imread(curr_image_path)
            curr_label = self.labels[curr_image_path]

            X[i,] = preprocess_image(curr_sample, self.dim[1], self.dim[0])

            y[i] = float(curr_label)


        # randomly flip half of batch
        flip_indices = np.random.choice(len(X), len(X)//2)

        X[flip_indices] = np.fliplr(X[flip_indices])
        y[flip_indices] = -y[flip_indices]

        return X, y