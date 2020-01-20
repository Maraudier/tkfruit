
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.constraints import maxnorm
#from keras.utils import to_categorical
import numpy as np
#import cv2

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
    
    #Conv2D(32, (3, 3), activation = 'relu', input_shape = (64, 64, 3)),
    #BatchNormalization(),
    #MaxPooling2D(pool_size = (2, 2)),
    #Conv2D(64, (3, 3), activation = 'relu'),
    #BatchNormalization(),
    #MaxPooling2D(pool_size = (2, 2)),
    #Conv2D(128, (3, 3), activation = 'relu'),
    #BatchNormalization(),
    #Flatten(),
    #Dense(128, activation = 'relu'),
    #Dropout(0.5),
    
    Dense(units = 6, activation = 'softmax')
])

#classifier = to_categorical(classifier, num_classes = 6)

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

training_set = train_datagen.flow_from_directory(
    "C:/Users/hanan/Desktop/GroupProject/FRFD/train", 
    target_size = (64, 64), 
    batch_size = 128, 
    class_mode = 'categorical'
)

test_set = test_datagen.flow_from_directory(
    "C:/Users/hanan/Desktop/GroupProject/FRFD/test", 
    target_size = (64, 64), 
    batch_size = 128, 
    class_mode = 'categorical'
)

# for multiprocessor use, add to fit_generator: use_multiprocessing = True, workers = x 
# x = your CPU Family. Run lscpu in linux to verify your available workers
classifier.fit_generator(
    training_set,
    steps_per_epoch = 100, 
    epochs = 30, 
    validation_data = test_set,
    validation_steps = 100
)

# Test predictions
classes = training_set.class_indices
print("Classes used: %s\n" % classes)
test_img = image.load_img("C:/Users/hanan/Desktop/GroupProject/orange.png")
test_img = image.img_to_array(test_img)
test_img = np.array(test_img, axis = 0)
result = classifier.predict(test_img)
print("Prediction result: %s\n" % result)
scores = classifier.evaluate(test_set, verbose = 0)
print("Accuracy: %.2f%%" % (scores[1] * 100)) 