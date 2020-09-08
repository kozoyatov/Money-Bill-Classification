from keras.applications.mobilenet_v2 import preprocess_input
from keras_preprocessing import image
import cv2
import numpy as np


def prepar_image(img):
    x = cv2.resize(img, (224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
