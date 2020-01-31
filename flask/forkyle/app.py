from flask import Flask, render_template, request
from model_init import Model

import tensorflow as tf
import base64

# Passing in Path name, image resize dimension, list of order the classification will be
# Declared here to allow initialization 
BF_mod = Model("BF_orig", 128, ['apple','orange','banana','rotten apple','rotten orange','rotten banana'])
IncV3_mod = Model("IncepV3", 299, ['rotten banana', 'banana', 'apple', 'rotten apple',
       'rotten orange', 'orange'])

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
    mapping = IncV3_mod.process_img(img)
    # --------------------------------------
    
    '''
    map2 = BF_mod.process_img(img)
    map3 = other_model.process_img(img)
    . . .
    UNUSED BUT SHOWS POTENTIAL TO ADD MORE MAPS
    '''

    b64_string = base64.b64encode(bytes)
    b64_data = "data:" + content_type + ";base64," + str(b64_string)[2:-1]

    return render_template("result.html", result=mapping, img_b64=b64_data)




app.run(debug=False)