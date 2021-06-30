# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I use the model as described in the research paper - (End to End Learning for Self-Driving Cars by NVIDIA). It has 5 convolution layers with relu activation and 3 fully connected layers. The output is the steering angle. The code for the model is given below.

```sh
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Conv2D(24,(5,5),strides=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(36,(5,5),strides=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(48,(5,5),strides=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
```

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and a separate set of data to turn smoothly in the corners. To create additional data and for better generalization I used mirror images of the same data. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model as detailed by NVIDIA in their research paper. I assumed this model will be appropriate since it is a tried and tested model,

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set with 80%, 20% split. To avoid overfitting we trained the model for fewer epochs (5).

The final step was to run the simulator to see how well the car was driving around track one. There was one point where the car was driving off the road. It was in the hard right hander towards the end. It was fixed by adding additional data to turn the car smoothly.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct its course.
Post testing for the first time I had to create an extra set of data for turning smoothly in the corners.
To augment the data sat, I also flipped images. I used the left and right camera image with steering correction 0.1 for additional data.

After the collection process, I had 32679 number of data points. I then preprocessed this data by normalizing to -0.5 to 0.5 and then cropping it so that we remove the dashboard part and the unnecessary environment part.
I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 and there was no major change in the accuracy after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
