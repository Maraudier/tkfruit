# Import the necessary Python libraries

from tf.keras import Sequential
from tf.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tf.keras.preprocessing.image import ImageDataGenerator
from tf.keras.preprocessing import image
from tf.keras.constraints import maxnorm
import numpy as np

# Define the structure of the model
classifier = Sequential([    
    Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu', padding = 'same'),
    Dropout(0.2),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
    MaxPooling2D(pool_size = (2, 2)),
    Dropout(0.3),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation = 'relu', padding = 'same'),
    Dropout(0.3),
    BatchNormalization(),
    Flatten(),
    Dropout(0.3),
    Dense(units = 128, activation = 'relu', kernel_constraint = maxnorm(3)),
    Dropout(0.3),
    BatchNormalization(),
    
    Dense(units = 6, activation = 'softmax')
])

# Compile the model and choose an optimizer
classifier.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy', 
    metrics = ['accuracy']
)

# Define the training data generator and specify data augmentation
train_datagen = ImageDataGenerator(
    rescale = 1./255, 
    shear_range = 0.2, 
    zoom_range = 0.2, 
    horizontal_flip = True
)

# Define the testing data generator
test_datagen = ImageDataGenerator(rescale = 1./255)

# Specify the source for the training data
training_set = train_datagen.flow_from_directory(
    "C:/Users/hanan/Desktop/GroupProject/FRFD/train", 
    target_size = (64, 64), 
    batch_size = 128, 
    class_mode = 'categorical'
)

# Specify the source for the testing data
test_set = test_datagen.flow_from_directory(
    "C:/Users/hanan/Desktop/GroupProject/FRFD/test", 
    target_size = (64, 64), 
    batch_size = 128, 
    class_mode = 'categorical'
)

# Printing the model's summary
print(classifier.summary())

# Fit the model
classifier.fit(
    training_set,
    epochs = 30, 
    validation_data = test_set,
    steps_per_epoch = 100,
    validation_steps = 100
)

# Evaluate the model & print its accuracy
classes = training_set.class_indices
scores = classifier.evaluate(test_set, verbose = 0)
print("Accuracy: %.2f%%" % (scores[1] * 100))