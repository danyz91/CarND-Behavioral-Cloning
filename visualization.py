
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import plot_model

import image_preprocessing
from model import load_partition
from model import build_model

def explore_dataset(labels):

    df = pd.DataFrame([labels])
    df = df.T
    axes = df.hist(bins=100, edgecolor='blue', linewidth=1.2)

    for ax in axes.flatten():
        ax.set_ylabel('# Elements in Dataset', {'fontweight': 'bold'})
        ax.set_xlabel('Steering wheel', {'fontweight': 'bold'})
        ax.set_title('Data Distribution Visualization', {'fontweight': 'bold'})
        ax.set_xlim((-1.0, 1.0))

    plt.show()

def main():

    # dataset params
    dataset_dir = '../simulator_data'

    # 1. Dataset info loading
    partition, labels = load_partition(dataset_dir)

    explore_dataset(labels)

    image_preprocessing.test_preprocessing(partition, labels)

    NVIDIA_INPUT_SHAPE = (66, 200, 3)

    model = build_model(NVIDIA_INPUT_SHAPE)
    plot_model(model, to_file='model.png', show_shapes=True)


if __name__ == '__main__':
    main()


