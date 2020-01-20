# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

"""
Created on Sat Jan 11 14:08:56 2020

@author: wkaht
"""

'''
DATASET FROM:
https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification

CNN Implementation
'''

import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
from tensorflow.keras import layers, models
import numpy as np
# import os #CURRENTLY UNUSED BECAUSE TENSORBOARD IS UNUSED
# import datetime
import pathlib

# Loading data set
data_dir = pathlib.Path('original_data_set')
image_count = len(list(data_dir.glob('*/*.png')))
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != '.ipynb_checkpoints'])

# The 1./255 is to convert from uint8 to float32 in range [0,1]. Split data into 80/20 
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

# CONSTANTS, currently unsure what to set IMG_HEIGHT and IMG_WIDTH
BATCH_SIZE = 128
IMG_HEIGHT = 256
IMG_WIDTH = 256
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     subset = 'training',
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode="sparse",
                                                     classes = list(CLASS_NAMES))
test_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     subset = 'validation',
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode="sparse",
                                                     classes = list(CLASS_NAMES))

# Splitting data generator into labels and images, currently unsure if this is best way to implement
train_images, train_labels = next(train_data_gen)
test_images, test_labels = next(test_data_gen)

# Model creation for CNN, currently unsure if 64 and 128 are good numbers to use and if layers are appropriate
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(.5),
    layers.Dense(6, activation='softmax')
])

# It's my understanding that these hyperparameters are good for our dataset
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# USING 10 EPOCHS for quicker processing on local PC
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels)) #,callbacks=[tensorboard_callback]
