# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 01:13:44 2020

@author: hanan
"""

# Import the Keras built-in dataset CIFAR10
from keras.datasets import cifar10

# Load the data & split it into the train & test sets
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

# Normalize the pixel values in the images to be between 0 and 1
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain = xtrain / 255.0
xtest = xtest / 255.0

# Import np_utils for creating one-hot representations of ytrain and ytest
from keras.utils import np_utils

# Create the one-hot representations of ytrain and ytest (as binary matrices)
ytrain = np_utils.to_categorical(ytrain)
ytest = np_utils.to_categorical(ytest)

# Import the Keras library needed for defining the input layer of the model
from keras.layers import Input

# Set the input layer's intended input shape
input_img = Input(shape = (32, 32, 3))

# Import the Keras libraries needed for defining the filters
from keras.layers import Conv2D, MaxPooling2D, concatenate

# Define the filters needed for the module
t1 = Conv2D(64, (1, 1), padding = 'same', activation = 'relu') (input_img) # 1x1 filter
t1 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu') (t1) # 3x3 filter

t2 = Conv2D(64, (1, 1), padding = 'same', activation = 'relu') (input_img) # 1x1 filter
t2 = Conv2D(64, (5, 5), padding = 'same', activation = 'relu') (t2) # 5x5 filter

t3 = MaxPooling2D((3, 3), strides = (1, 1), padding = 'same') (input_img) # 3x3 pooling
t3 = Conv2D(64, (1, 1), padding = 'same', activation = 'relu') (t3) # 1x1 filter

# Concatenate the 3 filters into the output of the module
output = concatenate([t1, t2, t3], axis = 3)

# Import the Keras libraries needed for flattening the output & 
# making a fully connected network for final classification
from keras.layers import Flatten, Dense

# Apply Flatten & Dense to the output
output = Flatten() (output)
out = Dense(10, activation = 'softmax') (output)

# Create the actual model
from keras.models import Model # Keras library needed for model creation

model = Model(inputs = input_img, outputs = out)
#print(model.summary())

# Apply SGD optimizer, compile & fit the model
from keras.optimizers import SGD
epochs = 25
lrate = 0.01
decay = lrate / epochs
sgd = SGD(lr = lrate, momentum = 0.9, decay = decay, nesterov = False)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
model.fit(xtrain, ytrain, validation_data = (xtest, ytest), epochs = epochs, batch_size = 32)

# Calculate accuracy of the model
scores = model.evaluate(xtest, ytest, verbose = 0)
print("Accuracy: %.2f%%" % (scores[1] * 100)) 