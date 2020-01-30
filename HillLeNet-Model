# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 06:19:41 2020

@author: Davis Hill
"""

# import packages
import tensorflow
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.utils import np_utils
from tensorflow.python.keras import backend as K
import keras.optimizers as op
from keras_preprocessing.image import ImageDataGenerator as IDG

import numpy as np

class LeNet:
  @staticmethod
  def build(lenet_version, lenet_functions, height, width, filters_first, filters_second, activation_hidden, activation_output, depth, num_labels):

    # initialize model according to image size
    model = Sequential()
    shape = (height, width, depth)

    if K.image_data_format() == "channels_first":
      shape = (depth, height, width)
    
    # first layer: 
    # filters of size 5x5
    # tanh/relu
    model.add(Conv2D(filters_first, 5, padding="same", input_shape=shape))
    model.add(Activation(activation_hidden))
    
    # older implementations of lenet utilize average pooling
    # modern implementations utilize max pooling
    # size 2x2 with stride of 2
    if(lenet_functions == 'original'):
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    else:
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second layer: 
    # filters of size 5x5
    # tanh/relu
    model.add(Conv2D(filters_second, 5, padding="same"))
    model.add(Activation(activation_hidden))
    
    # size 2x2 with stride of 2
    if(lenet_functions == 'original'):
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    else:
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
    # flatten to one dimensional column vector
    model.add(Flatten())
    
    # lenet versions four and five both utilized this additional fully connected layer
    # sigmoid/softmax
    if(lenet_version == 'four' or 'five'):
        model.add(Dense(120))
        model.add(Activation(activation_output))
    
    # lenet version five utilized this fully connected layer
    # sigmoid/softmax
    if(lenet_version == 'five'):
        model.add(Dense(84))
        model.add(Activation(activation_output))
        
    # lenet versions one, four, and five all utilized this full connected layer
    # sigmoid/softmax
    model.add(Dense(num_labels))
    model.add(Activation(activation_output))

    return model

# let user choose whether to run lenet model one, four or five
input_version = input("Input "'one'", "'four'", or "'five'" to specify the version of lenet you would like to run: ")

# when lenet was first implemented, the standard was to use tanh activation for hidden layers and sigmoid activation for output layer
# it is now standard to use relu activation for hidden layers and softmax activation for output layers
input_functions = input("Input either "'original'" or "'modern'" to specify the activation and pooling functions of Lenet model: ")

# dictionary that corresponds to model version
# index 0: height, 1: width, 2: number of filters in first convolutional layer, 3: numbers of filters in second convolutional layer
dict_version = {'one': (28, 28, 4, 8),
                'four': (32, 32, 4, 16),
                'five': (32, 32, 6 ,16)}

# dictionary that corresponds to modern vs original standards
dict_functions = {'original': ('tanh', 'sigmoid'),
                  'modern': ('relu', 'softmax')}

# transfer dictionary contents to tuple
tuple_version = dict_version[input_version]
tuple_functions = dict_functions[input_functions]

# build model
model = LeNet.build(input_version, 
                    input_functions, 
                    tuple_version[0], 
                    tuple_version[1], 
                    tuple_version[2], 
                    tuple_version[3], 
                    tuple_functions[0], 
                    tuple_functions[1], 
                    3, 6)

# preprocess data
datagen = IDG(rescale=1. / 255,
    rotation_range=10, 
	fill_mode='nearest',
    vertical_flip= True,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

#'/content/gdrive/My Drive/new/fruits-360/Training'
#'/content/gdrive/My Drive/new/fruits-360/Test'

# training generator
train_gen = datagen.flow_from_directory('C:/Users/Alfred/Downloads/freshrotten/dataset/train', 
	target_size=(tuple_version[0], tuple_version[1]), 
	color_mode='rgb',
	batch_size=32,
	class_mode="categorical",
	shuffle=True)

# testing generator
test_gen = datagen.flow_from_directory('C:/Users/Alfred/Downloads/freshrotten/dataset/test',
    target_size=(tuple_version[0], tuple_version[1]),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

# SGD optimizer
opt = op.SGD(lr=0.01)

# compile model
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["categorical_accuracy"])

# fit model
model.fit_generator(train_gen, epochs=30,
    validation_data=test_gen)
