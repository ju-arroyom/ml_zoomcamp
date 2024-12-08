from io import BytesIO
from urllib import request

from PIL import Image

import numpy as np
import tflite_runtime.interpreter as tflite


def get_model(path):
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def get_indices(interpreter):
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    return input_index, output_index

def predict(interpreter, X):
    input_index, output_index = get_indices(interpreter)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    return pred


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(img):
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = X*(1./255)
    X = np.array(X, dtype='float32')
    return X



def prediction(url):
    model_path = "model_2024_hairstyle_v2.tflite"
    img = download_image(url)
    img = prepare_image(img, (200, 200))
    X = preprocess_input(img)
    # Get Model
    interpreter = get_model(model_path)
    # Predict
    pred = predict(interpreter, X)
    float_prediction = pred[0].tolist()
    return {'output':float_prediction}


def lambda_handler(event, context):
    url = event["url"]
    result = prediction(url)
    return result
