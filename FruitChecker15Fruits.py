from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing import image
from keras.constraints import maxnorm
#from keras.utils import to_categorical
import numpy as np

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
    
    #Dense(units = 256, activation = 'relu', kernel_constraint = maxnorm(3)),
    #Dropout(0.3),
    #BatchNormalization(),
    
    Dense(units = 64, activation = 'relu', kernel_constraint = maxnorm(3)),
    Dropout(0.3),
    BatchNormalization(),
    
    Dense(units = 3, activation = 'softmax')
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
    "C:/Users/hanan/Desktop/GroupProject/FreshRottenFruitsDataset/train", 
    target_size = (64, 64), 
    batch_size = 128, 
    class_mode = 'categorical'
)

test_set = test_datagen.flow_from_directory(
    "C:/Users/hanan/Desktop/GroupProject/FreshRottenFruitsDataset/test", 
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
p = classifier.predict(test_set[:5])
print(np.argmax(p, axis = 1))

#test_image = image.load_img('', target_size = (64, 64))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = classifier.predict(test_image)
#training_set.class_indices

# Since we're using Softmax, this section may have to be changed to handle probabilities
#if result [0][0] == 1:
#    prediction = 'good'
#else:
#    prediction = 'bad'

#print(result, prediction)
