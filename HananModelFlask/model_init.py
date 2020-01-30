import tensorflow as tf

def init(model_dir, with_gpu = None):
    """Initialize the model and Tensorflow - called alongside the server startup script."""
    if with_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            except RuntimeError as e:
                print(e)
    
    return tf.saved_model.load(model_dir)

def preprocess_img(img):
    """Image preprocessing function to prepare image for inference in the model"""
    img.set_shape((None, None, 3))
    img = tf.image.resize(img, [64, 64])
    
    img = tf.reshape(img, (-1, 64, 64, 3))

    # Cast to float32 - 1st layer
    img = tf.cast(img, dtype = tf.float32)
    
    # Normalize data and invert
    img = (img / 255)

    return img

def value_mapper(inference_result):
    np_arr = inference_result[0].numpy()

    # The 6 classes that the image could possibly be classified into
    fruitclasses = ['Fresh apple','Fresh banana','Fresh orange','Rotten apple','Rotten banana','Rotten orange']

    mapping = {}

    for (key, value) in zip(fruitclasses, np_arr):
        mapping[key] = value.item()

    return mapping