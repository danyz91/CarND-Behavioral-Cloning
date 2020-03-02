
import os
import csv
import cv2
import numpy as np


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D
from sklearn.model_selection import train_test_split
import sklearn

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
    lines = list()

    with open(data_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = list()
    measurements = list()

    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = '../data/IMG'+filename
        curr_image = cv2.imread(current_path) # it is bgr, use scipy load image for rgb
        images.append(curr_image)
        curr_measurement = float(line[3])
        measurements.append(curr_measurement)

    '''

def load_dataset(dataset_dir, csv_filename='driving_log.csv', img_dir='IMG'):

    csv_path = os.path.join(dataset_dir, csv_filename)
    img_path = os.path.join(dataset_dir, img_dir)

    images = list()
    measurements = list()

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.2  # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
            path = img_path  # fill in the path to your training IMG directory

            filename_center = row[0].split('\\')[-1]
            filename_left = row[1].split('\\')[-1]
            filename_right = row[2].split('\\')[-1]


            img_center = mpimg.imread(os.path.join(img_path, filename_center))
            img_left = mpimg.imread(os.path.join(img_path, filename_left))
            img_right = mpimg.imread(os.path.join(img_path, filename_right))

            # add images and angles to data set
            images.append(img_center)
            images.append(img_left)
            images.append(img_right)

            measurements.append(steering_center)
            measurements.append(steering_left)
            measurements.append(steering_right)

    ## Data augmentation
    augmented_images = list()
    augmented_measurements = list()

    for image, measurement in zip(images, measurements):
        # Copy original dataset in new dataset
        augmented_images.append(image)
        augmented_measurements.append(measurement)

        # Flip curr image and its measurement
        image_flipped = np.fliplr(image)
        measurement_flipped = -measurement

        augmented_images.append(image_flipped)
        augmented_measurements.append(measurement_flipped)



    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)

    return X_train, y_train


def build_model(input_shape, loss='mse'):

    model = Sequential()

    #Normalization preproc via Lambda layer
    model.add(Lambda(lambda x: (x/255.0) - .5, input_shape=input_shape))

    #cropping
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    #Nvidia architecture
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))

    model.add(Dense(1))

    model.compile(loss=loss, optimizer='adam')

    return model

def generator(samples, batch_size=32):

    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def define_generator(samples):

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    generator(samples)

    # Set our batch size
    batch_size = 32

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    ch, row, col = 3, 80, 320  # Trimmed image format

    return train_generator, validation_generator



def trainiing_with_generator(model, train_generator, train_samples, validation_generator, validation_samples):

    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=
                                         validation_generator,
                                         nb_val_samples=len(validation_samples),
                                         nb_epoch=5, verbose=1)

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def main():

    # dataset params
    dataset_dir = '../simulator_data'

    # build params
    input_shape = (160, 320, 3)
    loss = 'mse'

    # fit params
    valid_split = 0.2
    epochs = 1

    # save params
    output_model_name = 'model.h5'

    X_train, y_train = load_dataset(dataset_dir)

    model = build_model(input_shape, loss=loss)

    model.fit(X_train, y_train, validation_split=valid_split, shuffle=True, nb_epoch=epochs)

    model.save(output_model_name)


if __name__=='__main__':
    main()