import csv
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

def fetch_path(source_path):
    folders = source_path.split('\\')
    filename = folders[-1]
    current_path = '../data/' + folders[-3] +'/IMG/' + filename
    return current_path

def importdata(lines, images, measurements):
    for line in lines:
        steering_center = float(line[3])
        
        source_path = line[0]
        current_path = fetch_path(source_path)
        image = plt.imread(current_path)
        
        images.append(image)
        images.append(np.fliplr(image))
        measurements.append(steering_center)
        measurements.append(-1*steering_center)
        
        source_path = line[1]
        current_path = fetch_path(source_path)
        image = plt.imread(current_path)
        
        images.append(image)
        images.append(np.fliplr(image))
        measurements.append(steering_center + correction)
        measurements.append(-1*(steering_center + correction))
        
        source_path = line[2]
        current_path = fetch_path(source_path)
        image = plt.imread(current_path)
        
        images.append(image)
        images.append(np.fliplr(image))
        measurements.append(steering_center - correction)
        measurements.append(-1*(steering_center - correction))
        
list = []
images = []
measurements = []
correction = 0.1

lines = []
with open('../data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
with open('../data/data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
with open('../data/data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
with open('../data/data4/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
importdata(lines, images, measurements)

npmeasurements = np.array(measurements)
npimages = np.array(images)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation
from keras.layers.convolutional import Conv2D

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
model.fit(npimages, npmeasurements, validation_split = 0.2, shuffle=True, epochs=5)

model.save('model1.h5')
