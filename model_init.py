import tensorflow as tf

def init(model_dir, with_gpu = True):
    if with_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            except RuntimeError as e:
                print(logical_gpus, e)
    
    return tf.saved_model.load(model_dir)



def prep_img(img):
    img.set_shape((None, None, 1))
    img = tf.image.resize(img, [150,150])

    img = tf.reshape(img, (-1, 150, 150, 1))


    img = tf.cast(img, (-1, 150, 150, 1))

    img = 1 - (img / 255)

    return img