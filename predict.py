import requests, sys
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf


class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = tf.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)


def preprocess(img,input_size):
    nimg = img.convert('RGB').resize(input_size, resample= 0)
    img_arr = (np.array(nimg))/255
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)


def predict_image(path_images: str, labels, model_path = 'model'):
    if model_path != "":
        model_path = model_path
    # Parameters
    input_size = (160,160)
    #define input shape
    channel = (3,)
    input_shape = input_size + channel
    MODEL_PATH = model_path+'/model.h5'
    model = load_model(MODEL_PATH,compile=False, custom_objects={'FixedDropout': FixedDropout})
    # read image
    im = Image.open(path_images)
    X = preprocess(im,input_size)
    X = reshape([X])
    y = model.predict(X)
    return {
       "result": labels[np.argmax(y)],
       "weighted": np.max(y)
    }

labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
result = predict_image(sys.argv[1], labels, sys.argv[2])
print(result)