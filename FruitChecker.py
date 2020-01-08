
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.util import to_categorical
import numpy as np

#Folder named original_data_set contains at least 1 folder of good and 1 folder of bad fruit
dataLocation = 'original_data_set'

classifier = Sequential([
    Flatten(), # flattens the input to the model
    Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'),
    Conv2D(64, (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    Dense(units = 128, activation = 'relu'),
    Conv2D(128, (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    Dense(units = 1, activation = 'softmax')
])

classifier = to_categorical(classifier, num_classes = 6)

classifier.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy', 
    metrics = ['accuracy']
)

train_datagen = ImageDataGenerator(
    rescale = 1./255, 
    shear_range = 0.2, 
    zoom_range = 0.2, 
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory(
           dataLocation, 
           target_size = (64, 64), 
           batch_size = 32, 
           class_mode = 'binary'
)

training_set = train_datagen.flow_from_directory(
           dataLocation, 
           target_size = (64, 64), 
           batch_size = 32, 
           class_mode = 'binary'
)

#numSteps takes the total of images and divides by 32 (because... Math). Modify this if your image set is larger or smaller accordingly
numSteps = 3000/32
#for multiprocessor use, add to fit_generator: use_multiprocessing = True, workers = x 
# x = your CPU Family. Run lscpu in linux to verify your available workers
classifier.fit_generator(training_set, steps_per_epoch = numSteps, epochs = 25, validation_data = test_set, validation_steps = numSteps)

test_image = image.load_img('fruitcheck3.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
#if result [0][0] == 1:
#    prediction = 'good'
#else:
#    prediction = 'bad'

#print(result, prediction)
