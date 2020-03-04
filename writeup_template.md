## Project: Behavioral Cloning 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<br/><br/>

# Overview

In this project, there is shown how a deep neural network can be trained on expert driving in order to drive a car along a track. The test is run in a simulator. This is the implementation of an end-to-end network that learns to map driving images into driving actions, in this case steering angles.


The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


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


<br/><br/>


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

- Python script to create and train the model
    :[ model.py](./model.py)
- Python script for driving the car in autonomous mode
    :[ drive.py](./drive.py)
- Hierarchical Data Format (.h5) file with trained neural network
    :[ model.h5](./model.h5)
- A writeup report 
    You are here! -> :[ Writeup](./README.md)


#### 2. Drive autonomously
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. visualization.py and data_generation.py

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Training Strategy

#### 1.Creation of the Training Set 

- Analog stick
- Driving on both tracks
- Driving backwards
- Steering Correction for multiple cameras
- Data augmentation
- Flipping

![alt text][image1]


#### 2. Data Preprocessing

![alt text][image8]
![alt text][image9]

#### 3. Data Augmentation

* Original Samples
![alt text][image3]

* Brightened Samples
![alt text][image4]

* Vertical Shifted
![alt text][image6]

* Random Shadowed Images
![alt text][image5]


### Model Architecture

#### 1. Model Architecture

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 


The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)
![alt text][image2]


#### 2. Dropout

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


### Training Results

![alt text][image7]


### Results

- Video final low resolution


    :<iframe width="560" height="315" src="https://www.youtube.com/embed/DGrpKJIvUVo" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

- Video final high resolution

    :<iframe width="560" height="315" src="https://www.youtube.com/embed/aV5HmHDBOKA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

- epoch1-3-6 on test track


Training Start 
<iframe width="560" height="315" src="https://www.youtube.com/embed/WOTVHgoLYyg" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> 

Training End
<iframe width="560" height="315" src="https://www.youtube.com/embed/tMMu3ZyTAak" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



- Maneuver recovery

    :<iframe width="560" height="315" src="https://www.youtube.com/embed/WSVCbDsjtoY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>






### Known Issues and Open Points

- Jungle test track
- Add other augmentations

