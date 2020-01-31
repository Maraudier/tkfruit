import tensorflow as tf

class Model:
    def __init__(self, fold_name, resize, class_list):
        self.fold_name = fold_name
        self.resize = resize
        self.model_dir = "./models/" + self.fold_name
        self.class_list = class_list
        # Loading the saved model from directory
        self.model = tf.saved_model.load(self.model_dir)
        self.sig = self.model.signatures["serving_default"]
        
       
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
        
        # Putting image into model and mapping result to appropriate values
        mapping = self.__map_result(self.model(img))
    
        return mapping
    
    # ONLY used within process_img, maps model results
    def __map_result(self, result):
        np_arr = result[0].numpy()

        # 6 values which represent what fruits our model classifies
        chars = self.class_list
    
        mapping = {}
    
        for (key, value) in zip(chars, np_arr):
            mapping[key] = value.item()
    
        return mapping