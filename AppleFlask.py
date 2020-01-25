from flask import Flask, render_template, request
from model_init import init, prep_img

import tensorflow as tf

model = init("GoodApple")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def inference():
    if not request.files["img"]:
        return "No image uploaded"

    # Load image from multipart upload and get bytes
    file = request.files["img"]
    bytes = file.read()

    # Convert raw bytes to a tensor
    img = tf.image.decode_image(bytes, channels=1)
    img = prep_img(img)

    predict = model.predict()

    return render_template("result.html", result=predict)

app.run(debug=True)