
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random


def preprocess_image(image, resize_width=200, resize_height=66, augment=False):
    """
    Apply preprocessing pipeline to input image

    :param image: input image
    :param resize_width: target width
    :param resize_height: target height
    :param augment: a bool signaling if data augmentation must be set or not
    :return: image with normalization, cropping and resize applied. It is also augmented if selected
    """
    crop_region_low = 50
    crop_region_high = -25

    if augment:
        bool_values = [True, False]

        if random.choice(bool_values):
            image = augment_brightness_camera_images(image)
        if random.choice(bool_values):
            image = trans_image(image)
        if random.choice(bool_values):
            image = add_random_shadow(image)

    image = normalize(image)
    image = crop_horizontally(image, crop_region_low, crop_region_high)
    image = resize(image, resize_width, resize_height)

    return image


def normalize(image):
    """
    Apply normalization. Divide by 255 and then subtract -5
    :param image: input image
    :return: normalized image
    """
    out_image = (image / 255.0) - .5
    return out_image


def crop_horizontally(image, low, high):
    """
    Crops the image along x dimension
    :param image: input image
    :param low: value from which the selection start
    :param high: value to which the selection ends
    :return: cropped version of input image
    """
    out_image = image[low:high,:,:]
    return out_image


def resize(image, width, height):
    """
    Resize the image at given width and height
    :param image: input image
    :param width: target width
    :param height: target height
    :return:
    """
    out_image = cv2.resize(image, (width, height), cv2.INTER_AREA)
    return out_image


def augment_brightness_camera_images(image):
    """
    Apply a random brightness adjustment to input image. It works on Hue channel of HSV version of the image
    :param image: input image
    :return: a randomly brightened version of image
    """

    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def trans_image(image):
    """
    Transpose vertically the image by a random value. The remaining part is filled with 0 pixels
    :param image: input image
    :return: a vertical transposed version of the image
    """

    # Translation
    tr_x = 0
    tr_y = 40 * np.random.uniform() - 40 / 2
    trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    height, width = image.shape[:2]

    image_tr = cv2.warpAffine(image, trans_M, (width, height))

    return image_tr


def add_random_shadow(image):
    """
    Apply a random shadow to the image. It works on Hue Lightness and Saturation version of image
    :param image: input image
    :return: a randomly shadowed version of input image
    """

    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1

    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2)==1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1]*random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0]*random_bright

    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    return image


def test_brightness(samples, labels):
    """
    Test brightness step
    :param samples: test samples
    :param labels: test labels
    :return:
    """

    fig, axes = plt.subplots(1, len(samples))
    for ind, sample in enumerate(samples):
        bright_image = augment_brightness_camera_images(sample)
        axes[ind].imshow(bright_image)
        axes[ind].set_title('steer: {:.2f}'.format(labels[ind]))


def test_trans(samples, labels):
    """
    Test transposing step
    :param samples: test samples
    :param labels: test labels
    :return:
    """

    fig, axes = plt.subplots(1, len(samples))
    for ind, sample in enumerate(samples):
        tr_image = trans_image(sample)
        axes[ind].imshow(tr_image)
        axes[ind].set_title('steer: {:.2f}'.format(labels[ind]))


def test_shadow(samples, labels):
    """
    Test shadowing step
    :param samples: test samples
    :param labels: test labels
    :return:
    """
    fig, axes = plt.subplots(1, len(samples))
    for ind, sample in enumerate(samples):
        cropped = crop_horizontally(sample, 50, -25)
        axes[ind].imshow(cropped)
        axes[ind].set_title('steer: {:.2f}'.format(labels[ind]))


def test_crop(samples, labels):
    """
    Test crop step
    :param samples: test samples
    :param labels: test labels
    :return:
    """
    fig, axes = plt.subplots(1, len(samples))
    for ind, sample in enumerate(samples):
        shadowed = add_random_shadow(sample)
        axes[ind].imshow(shadowed)
        axes[ind].set_title('steer: {:.2f}'.format(labels[ind]))


def test_preprocessing(partition, labels):
    """
    Test all preprocessing step
    :param partition: a dictionary containing all images path
    :param labels: a dictionary containing mapping between paths and labels
    :return:
    """
    TEST_SIZE = 5

    training_samples = partition['train']

    test_samples_paths = random.sample(training_samples, TEST_SIZE)

    test_samples = list()
    test_labels = list()

    for path in test_samples_paths:
        test_samples.append(mpimg.imread(path))
        test_labels.append(labels[path])

    fig, axes = plt.subplots(1, TEST_SIZE)
    for ind, sample in enumerate(test_samples):
        axes[ind].imshow(sample)
        axes[ind].set_title('steer: {:.2f}'.format(test_labels[ind]))

    test_brightness(test_samples, test_labels)
    test_trans(test_samples, test_labels)
    test_shadow(test_samples, test_labels)
    test_crop(test_samples, test_labels)


    plt.show()





