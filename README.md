## Project: Behavioral Cloning 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview

In this project, there is shown how a deep neural network can be trained on expert demonstrations in order to drive a car along a track. The test is run in a simulator.

This is the implementation of an end-to-end network that learns to map driving images into driving actions, in this case steering angles.


The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./img/dataset_visualization.png "Dataset Histogram"
[image2]: ./img/model.png "Model"
[image3]: ./img/original_img.png "Dataset samples"
[image4]: ./img/brightened.png "Preprocessing: Brightening"
[image5]: ./img/shadowed.png "Preprocessing: Shadowing"
[image6]: ./img/vertical_shifted.png "Preprocessing: Vertical Shifting"
[image7]: ./img/train_valid_loss.png "Training and Validation Loss"
[image8]: ./img/pre_crop.png "Pre Cropped Images"
[image9]: ./img/crop.png "Cropped Images"

[image10]: ./img/cameras_view.png "Cameras View"
[image11]: ./img/correction.png "Camera correction needed"


[image12]: ./img/unflipped.png "Unflipped Image"
[image13]: ./img/flipped.png "Flipped Image"


<br><br/>

### Submission
---

#### 1. Requested files

The files requested for the submission are listed below:

- Python script to create and train the model  
    [ model.py](./model.py)

- Python script for driving the car in autonomous mode 
    [ drive.py](./drive.py)

- Hierarchical Data Format (.h5) file with trained neural network
    [ model.h5](./model.h5)

- Video file showing a complete lap on test track
    [ video.mp4](./video.mp4)

- A writeup report 
    You are here! -> [Writeup](./README.md)


#### 2. Additional Files and Updates

Besides requested files, in the repo there are other two files:

- **data_generation.py**

    This file contains ```DataGenerator``` class which inherits from ```keras.utils.Sequence```. 
    This class can be used as batch generator in ```fit_generator``` class. 
    This is done in order to generate data for training on-the-fly rather tan storing the entire dataset in memory. Inspiration for such code organization came from [this Stanford's team blog post](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
    The template here provided has been extended with *ad hoc* needs of this specific problem, as discussed below.

- **visualization.py**

    This script contains all code useful for display dataset and to plot preprocessing informations.
    This is useful to generate image information without executing the main script contained in ```model.py``` file.
    The result of the execution of this script is:

    + Dataset histogram visualization
    + Preprocessing step visualization
    + Model image saving in *model.png* file

- **image_preprocessing.py**

    This script contains all the code for image preprocessing. 
    Preprocessing pipeline is described below in this file.
    This script contains also testing function to plot pipeline intermediate stages.

- **drive.py**

    The provided script has been slightly modified so it can also support *.hdf5* file format for intermediate weight visualization.
    If a hdf5 file is specified via command line, the script build the model contained in *model.py*'s ```build_model``` function and then load the specified weights.
    This has been done in order to visualize how intermediate stages of training affect actual driving


The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline used for training and validating the model, and it contains comments to explain how the code works.

#### 3. Drive autonomously

Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

<br><br/>
### Training Strategy
---

#### 1. Creation of the Training Set 

In order to create a good dataset, different strategies have been used. They are listed and detailed below.

All data have been recorded running the simulator at its lowest configuration:

- Resolution: 640x480
- Graphics Quality: 'Fastest' 

This give us also the chance to prove generalization of our training over better image at higher resolution. Indeed, images recorded at this resolution have more effect in shadowing and light conditions, which will be unseen for our network.

##### 1.1 Joystick Driving

In order to provide smooth steering signals, I have recorded different laps using the following setup:

-  [Linux Udacity Simulator v1](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
- Xbox 360 Joystick

In this way the recorded signal is smoother than recording keyboard arrows and it is easier to drive than mouse input handling.

##### 1.2 Driving on both tracks

First laps have been recorded by driving on first track.

In order to augment the driving situation seen by our network during training, there have been recorded some laps also on second track of version 1 of Udacity Simulator.

##### 1.3 Driving backwards

First laps have been recorded by driving on first track clock-wise.

After some considerations about data biasing over left turns, other laps have been recorded **counter-clockwise**. 

##### 1.4 Driving from side to center

It has been driven video chunks where the driver goes from the road side to lane center.

This has been done in order to give examples of position recovering to lane center.  

##### 1.5 Steering Correction for multiple cameras

In order to use all data recorded during previous recording stages, there have been loaded into dataset also the side cameras take from left and right point of view showed below.

![alt text][image10] 

Taking these images as input raises the problem of correcting the steering signal recorded in order to augment data with correct signals.

The problem we need to face is showed in the image below.

![alt text][image11] 


It has been chosen to apply a correction of ```+/- 0.2``` to steering signal based on center camera image.

#### 2. Dataset Visualization

##### Random Samples

Here there are provided some random samples from the dataset with the corresponding steering wheel.

Original Samples
![alt text][image3]

In these sample we can see:

* Center images with low steering wheel value
* Left-side images with positive steering correction
* Right-side images with negative steering correction


##### Dataset Histogram

After all the considerations listed above, the complete recorded dataset distribution is showed in the image below.

![alt text][image1]

From the inspection of the image, it is clear that:

* The dataset is biased towards center driving.
* Some turns angles are more frequent than other.
* The dataset is quite symmetric to center driving and this is a good measurement that driving backwards helped a lot in this balancing.
* Some data augmentation could help generalization


#### 3. Data augmentation

Some data augmentation technique have been used. Some of them have been inspired by [this guide](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.d779iwp28)

The code for this functions is contained in ```image_preprocessing.py``` file.

##### Brightening
---
Randomly, some images are made brighter or darker using HSV color space.

Here it is the result on sample images:

![alt text][image4]

##### Vertical Shifting
---
Randomly, some images are vertically shifted and filled with black pixels.

It has been chosen to vertical shift only in order to not modify their relative steering value.

Here it is the result on sample images:

![alt text][image6]

##### Random Shadowing
---
Randomly, some images are shadowed with a random shaped shadow. This shadow is computed over HLS color space.

Here it is the result on sample images:

![alt text][image5]


##### Flipping
---
In order to balance and augment road situations, in each batch images are flipped randomly.

Original Image | Flipped Image
--- | ---
![alt text][image12] | ![alt text][image13]

<br><br/>
### Model Architecture
---

#### 1. Preprocessing

All data fed to neural network are :

* **Normalized**
    This is done in order to help stability during training. 

    The code is at **line 32** of ```image_preprocessing.py``` file

* **Cropped**
    This is done in order to remove useless informations from the image, such as sky and trees in the background. 

    The code is at **line 33** of ```image_preprocessing.py``` file.

    The cropping result is showed below

    ![alt text][image8]

    ![alt text][image9]

* **Resized**
    All images are resized to have width and height of (200,66).

    This is done in order to match with [NVIDIA Architecture](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

    The code is at **line 34** of ```image_preprocessing.py``` file



#### 2. Model Architecture

The model chosen for completing the task is the NVIDIA End-To-End architecture presented in the paper at this [link](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

This is chosen particularly because this net has been used for driving a real car within the same problem formulation.

The network architecture has 11 layers and it is showed below.
| Layer                 |   Info
|:---------------------:|---------
| Input                 |                
| Convolution 5x5  + ReLU     | 24 Kernels
| Convolution 5x5  + ReLU     | 36 Kernels
| Convolution 5x5  + ReLU     | 48 Kernels
| Convolution 3x3  + ReLU     | 64 Kernels
| Convolution 3x3  + ReLU     | 64 Kernels
| Flatten           |
| Fully Connected       |  100 neurons
| Fully Connected       |  50 neurons
| Fully Connected       |  10 neurons 
| Fully Connected       |  1 neuron





#### 3. Reduce Overfitting

The model contains two dropout layers in order to reduce overfitting (model.py lines 74 and line 78). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. For this the ```train_test_split``` from scikit-learn library has been used (code line 56).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 85).

<br></br>
### Final Architecture
---
![alt text][image2]


<br></br>
### Training Results
---

The loss history on training set and validation set is showed in the image below.

![alt text][image7]

From the inspection of the image it is clear that the model does:

* No underfitting, since performance gets better over epochs on training set
* No overfitting, since validation loss decreases over epochs


<br></br>
### Video Results
---

Below, there are listed video sources of some driving situations encountered.


**Video final low resolution**

The video presents result in completing test track autonomously

:[![final](https://img.youtube.com/vi/DGrpKJIvUVo/0.jpg)](https://www.youtube.com/watch?v=DGrpKJIvUVo) 



**Generalization to higher resolution**

The video presents how the car manages to handle unseen resolutions and shadows present in higher resolution video.

:[![final](https://img.youtube.com/vi/aV5HmHDBOKA/0.jpg)](https://www.youtube.com/watch?v=aV5HmHDBOKA) 
  

**Improvements on 2nd Track**

The video shows how the vehicle sub-perform over 2nd track after **1 epoch** of training

:[![final](https://img.youtube.com/vi/WOTVHgoLYyg/0.jpg)](https://www.youtube.com/watch?v=WOTVHgoLYyg) 

At the end of training the car manages to drive along 2nd track.

:[![final](https://img.youtube.com/vi/tMMu3ZyTAak/0.jpg)](https://www.youtube.com/watch?v=tMMu3ZyTAak) 


**Maneuver recovery**

A possible use case of such neural network can also be **maneuver recovery**.

In the video, the human driver injects some dangerous situations while the car is driving autonomously.

This is done in order to test whether the car would manage to revert to lane center.

In the video below when you read 
```diff
Mode:Manual
```
the user is controlling the vehicle and you can see that the car manages to revert to center lane.


:[![final](https://img.youtube.com/vi/WSVCbDsjtoY/0.jpg)](https://www.youtube.com/watch?v=WSVCbDsjtoY) 


<br></br>
### Known Issues and Open Points
---

- **Z Order Handling**
    :The model needs to generalize better to unseen situations, in particular for climbs and descents. These situations are not present in the dataset recorded and it is where the net suffer the most

- **Dataset balancing**
    : Generalization can be better achieved by balancing the dataset with augmentation techniques aiming in representing different turning signals