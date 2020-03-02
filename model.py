
import os
import csv
import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
import sklearn
import math
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from data_generation import DataGenerator
import image_preprocessing

def load_partition(dataset_dir, csv_filename='driving_log.csv', img_dir='IMG'):

    csv_path = os.path.join(dataset_dir, csv_filename)
    img_path = os.path.join(dataset_dir, img_dir)

    images = list()

    partition = dict()
    labels = dict()

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            correction = .2  # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
            filename_center = row[0].split('\\')[-1]
            filename_left = row[1].split('\\')[-1]
            filename_right = row[2].split('\\')[-1]

            img_center = os.path.join(img_path, filename_center)
            img_left = os.path.join(img_path, filename_left)
            img_right = os.path.join(img_path, filename_right)

            # add images and angles to data set
            images.append(img_center)
            images.append(img_left)
            images.append(img_right)

            labels[img_center] = float(steering_center)
            labels[img_left] = float(steering_left)
            labels[img_right] = float(steering_right)

    train_samples, validation_samples = train_test_split(images, test_size=0.2)

    partition['train'] = train_samples
    partition['validation'] = validation_samples

    return partition, labels


def build_model(input_shape, loss='mse', learning_rate=.001):

    model = Sequential()

    #Nvidia architecture
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu',input_shape=input_shape))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))

    model.add(Dense(1))

    model.compile(loss=loss, optimizer=Adam(lr=learning_rate))

    return model


def main():

    # dataset params
    dataset_dir = '../simulator_data'

    # build params
    n_channels = 3
    INPUT_IMAGE_SHAPE = (160, 320, n_channels)

    NVIDIA_INPUT_SHAPE = (66, 200, n_channels)

    loss = 'mse'
    learning_rate = 0.001

    # save params
    output_model_name = 'model.h5'

    # Steps

    # Parameters
    params = {'dim': (NVIDIA_INPUT_SHAPE[0], NVIDIA_INPUT_SHAPE[1]),
              'batch_size': 32,
              'n_channels': n_channels,
              'shuffle': True}

    partition, labels = load_partition('../simulator_data')

    # test preprocessing
    #image_preprocessing.test_preprocessing(partition, labels)

    # Generators
    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)

    model = build_model(NVIDIA_INPUT_SHAPE, loss=loss, learning_rate=learning_rate)

    filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=5)

    # Train model on dataset
    history_object = model.fit_generator(generator=training_generator, validation_data=validation_generator,
                                         steps_per_epoch=len(training_generator),
                                         validation_steps=len(validation_generator), use_multiprocessing=True,
                                         workers=8, nb_epoch=7,  verbose=1, callbacks=[checkpoint, stopper])

    ### print the keys contained in the history object
    print(history_object.history.keys())
    print(history_object.history)

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    model.save(output_model_name)


if __name__ == '__main__':
    main()
