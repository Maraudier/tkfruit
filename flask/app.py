from flask import Flask, render_template, request
from model_init import Model, map_result

import tensorflow as tf
import numpy as np
import base64

# Passing in Path name, image resize dimension, list of order the classification will be
# Declared here to allow initialization 
BF_mod = Model("BF_orig", 128, 'pb')
resNet_mod = Model("resnet.h5", 64, 'h5')
vgg16_mod = Model("vgg16", 224, 'pb')
hanan_mod = Model("HananModel", 64, 'pb')
v3matt_mod = Model("v3inceptionMatt", 299, 'pb')
lenet_mod = Model("lenet.h5", 32 ,'h5')


app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def inference():
    if not request.files["img"]:
        return render_template("noImg.html")
    
    # Load image from upload and put into bytes
    file = request.files["img"]
    content_type = file.content_type
    bytes = file.read()

    # bytes to a tensor
    img = tf.image.decode_image(bytes, channels=3)
    
    # RIGHT NOW THIS IS THE ONLY MODEL THAT GETS DISPLAYED
    BF = BF_mod.process_img(img)
    resnet = resNet_mod.process_img(img)
    vgg = vgg16_mod.process_img(img)
    hanan = hanan_mod.process_img(img)
    V3 = v3matt_mod.process_img(img)
    lenet = lenet_mod.process_img(img)  # LENET OUTPUTS SIGMOID SO VALUES HAVE TO BE ALTERED
    len_maxind = np.argmax(lenet)
    n_lenet = np.zeros(lenet.size)
    n_lenet[len_maxind] = 1
    # --------------------------------------

    # Ensemble adds the numpy results of each model
    # NOTE: different indices are used based on the order 
    #       in which models were classified during training
    ensemble = [BF[0] + V3[0] + vgg[2] + hanan[0] + resnet[0] + n_lenet[0],
                BF[2] + V3[1] + vgg[1] + hanan[1] + resnet[1] + n_lenet[1],
                BF[1] + V3[2] + vgg[5] + hanan[2] + resnet[2] + n_lenet[2],
                BF[3] + V3[3] + vgg[3] + hanan[3] + resnet[3] + n_lenet[3],
                BF[5] + V3[4] + vgg[0] + hanan[4] + resnet[4] + n_lenet[4],
                BF[4] + V3[5] + vgg[4] + hanan[5] + resnet[5] + n_lenet[5]]
    ensemble = np.multiply(ensemble,100)
    ensemble = np.divide(ensemble,6)
    infer = map_result(ensemble)
    # infer = map_result(n_lenet)

    # Converting bytes to image-readable data for HTML 
    b64_string = base64.b64encode(bytes)
    b64_data = "data:" + content_type + ";base64," + str(b64_string)[2:-1]
    
    # Handle multiple buttons, depending on graphical or data needs
    if request.form['action'] == 'Inference Data':
        return render_template("infer.html", result=infer, img_b64=b64_data)
    elif request.form['action'] == 'Submit':
        result = max(infer, key=infer.get)
        return render_template("result.html", result=result, img_b64=b64_data)
    else:
        return 'Bad request!', 400



if __name__ == "__main__":
    app.run(debug=False)