import tensorflow as tf

class Model:
    def __init__(self, filepath, resize, save_type):
        self.fold_name = filepath
        self.resize = resize
        self.model_dir = "./models/" + self.fold_name
        # Loading the saved model from directory
        if (save_type == 'pb'):
            self.model = tf.saved_model.load(self.model_dir)
        elif (save_type == 'h5'):
            self.model = tf.keras.models.load_model(self.model_dir)
        

    # Preprocess and run image through model, returns a map to results
    def process_img(self, img):
        """Image preprocessing function to prepare image for inference in the model"""
        img.set_shape((None, None, 3))
        
        # ADJUST SIZE PARAMETERS BASED ON MODEL
        img = tf.image.resize(img, [self.resize, self.resize])
        
        img = tf.reshape(img, (-1, self.resize, self.resize, 3))
    
        # Casting to float32 - necessary as this is the input type of my models 
        # first layer
        img = tf.cast(img, dtype=tf.float32)
        
        # Normalize data
        img = img / 255
        
        # Putting image into model
        result = self.model(img)
    
        # Convert result to a numpy array
        return result[0].numpy()
    
# Converts numpy_array results into mapped data with labels
def map_result(np_arr):

    # 6 values which represent what fruits our model classifies
    chars = ['Apple', 'Banana', 'Orange', 'Rotten Apple', 'Rotten Banana', 'Rotten Orange']

    mapping = {}

    for (key, value) in zip(chars, np_arr):
        mapping[key] = value.item()

    return mapping