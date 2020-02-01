from flask import Flask, render_template, request
#from model_init import Model

import tensorflow as tf
import base64

# ONLY used within process_img, maps model results
def map_result(result):
    
    # 6 values which represent what fruits our model classifies
    chars = ['apple', 'banana', 'orange', 'rotten apple', 'rotten banana', 'rotten orange']
  
    mapping = {}
    
    for (key, value) in zip(chars, result):
        mapping[key] = value.item()
    
    return mapping

class Model:
    def __init__(self, fold_name, resize):
        self.fold_name = fold_name
        self.resize = resize
        self.model_dir = "./" + self.fold_name
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
        
        result = self.model(img)
        
        return result[0].numpy()
    #   Putting image into model and mapping result to appropriate values
    #   mapping = self.__map_result(self.model(img))
    
    #   return mapping

# Passing in Path name, image resize dimension, list of order the classification will be
# Declared here to allow initialization 
BF_mod = Model("BF_orig", 128)
IncV3_mod = Model("IncepV3", 299)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def inference():
    if not request.files["img"]:
        return "No image uploaded"
    
    # Load image from upload and put into bytes
    file = request.files["img"]
    content_type = file.content_type
    bytes = file.read()

    # bytes to a tensor
    img = tf.image.decode_image(bytes, channels=3)
    
    # RIGHT NOW THIS IS THE ONLY MODEL THAT GETS DISPLAYED
    BF = BF_mod.process_img(img)
    V3 = IncV3_mod.process_img(img)
    ensemble = [BF[0] + V3[2],
                BF[2] + V3[1],
                BF[1] + V3[5],
                BF[3] + V3[3],
                BF[5] + V3[0],
                BF[4] + V3[4]]
    # --------------------------------------
    
    
    '''
    map2 = BF_mod.process_img(img)
    map3 = other_model.process_img(img)
    . . .
    UNUSED BUT SHOWS POTENTIAL TO ADD MORE MAPS
    '''

    result = map_result(ensemble)

    b64_string = base64.b64encode(bytes)
    b64_data = "data:" + content_type + ";base64," + str(b64_string)[2:-1]

    return render_template("result.html", result=result, img_b64=b64_data)




app.run(debug=False)