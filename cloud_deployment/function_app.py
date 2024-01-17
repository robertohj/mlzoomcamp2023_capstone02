import azure.functions as func
import logging
# Model imports (notice NO tensorflow)
import pickle
import numpy as np
from PIL import Image
import tensorflow.lite as tflite
import requests
import json

# Constants
LARGE_IMAGE_SIZE = 224  # Size that the base model has as default input size
IMG_SIZE = (LARGE_IMAGE_SIZE,LARGE_IMAGE_SIZE)


# preprocessing without Tensorflow
# See: https://github.com/qubvel/efficientnet/blob/master/efficientnet/model.py
#  and https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L18
#if mode == 'torch':
#        x /= 255.
#        mean = [0.485, 0.456, 0.406]
#        std = [0.229, 0.224, 0.225]
def preprocess_input(x):
    x /= 255.
    #mean = [0.485, 0.456, 0.406]
    #std = [0.229, 0.224, 0.225]
    return x

# load_img >> this replaces the load_image from the tf/keras packages
def load_img(img_url, img_size):
    #img_path: path including file name
    # img_size: tuple, e.g. (LARGE_IMAGE_SIZE,LARGE_IMAGE_SIZE)
    # THIS VERSION handles retrieval from an url (needed for azure/cloud/aws functions)
    # REFERENCE: https://stackoverflow.com/questions/7391945/how-do-i-read-image-data-from-a-url-in-python

    


    # with Image.open(img_url) as img:  # if reading a local file
    with Image.open(requests.get(img_url, stream=True).raw) as img: # if reading from url
        img = img.resize(img_size, Image.NEAREST)
        x = np.array(img, dtype = 'float32')
        return x

# Load the TFLite model (converted from the keras model)
interpreter = tflite.Interpreter(model_path = "EfficientNetB3_large_07_0.927.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Load the classes
# The classes were also saved previously as pkl, so we can use them as such:
with open('encoded_classes.pkl', 'rb') as inp:
    encoded_classes = pickle.load(inp)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
@app.route(route="predict")
def predict(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Predicting...")
    
    #Get parameter (img path)
    #img = "./img/African Painted Dogs 0187 - Grahm S. Jones, Columbus Zoo and Aquarium.jpg"
    # IMPORTANT: for url images, they need to be sent in a proper way, 
    # for instance, the following url
    #   http:// https://www.timeforkids.com/wp-content/uploads/2023/11/G3G5_231117_bear_steps.jpg
    # must be sent as  ...predict?img=img=https%3A%2F%2Fwww.timeforkids.com%2Fwp-content%2Fuploads%2F2023%2F11%2FG3G5_231117_bear_steps.jpg
    # tip: check out the formating in https://www.urlencoder.org/
    # Additionally, the url can be encoded in python like described here: https://www.urlencoder.io/python/
    # use       urllib.parse.urlencode(params)
    # 
    img = req.params.get('img')
    if not img:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            img = req_body.get('img')

    if img:
        # Load the images using the new load_img() function already returns an np.array(dtype='float32')
        # For functions in the cloud, this version of load_img handles the url retrival
        #logging.info("\nImage: ",img)  # WARNING: if attempting to print the whole string, will throw an error due to formatting
        x = load_img(img, IMG_SIZE)
        X = preprocess_input(x)
        X = np.expand_dims(X, 0) # expand dimensions (make it "batch")
        #X = np.array([X])  # expand dimensions
        
        # This was using the keras model:
        #pred = best_EfficientNetB3_model.predict(X)
        # This is actual predictions using the tflite model:
        interpreter.set_tensor(input_index,X)
        interpreter.invoke()
        pred_tflite = interpreter.get_tensor(output_index)
        # The rest is the same:
        y_pred = np.argmax(pred_tflite, axis = 1)
        predicted_class = y_pred
        predicted_proba = pred_tflite.flatten()[y_pred][0]
        predicted_label = encoded_classes[y_pred[0]]
        logging.info(f"Predicted class: {predicted_class}")
        logging.info(f"Predicted label: {predicted_label}")
        logging.info(f"Predicted proba: {predicted_proba}")

        logging.info("Done.")

        result = {
            "class_label": predicted_label,
            "class_probability": float(predicted_proba)
        }

        # # convert (serialize) dictionary to JSON string object
        json_string = json.dumps(result)
        return func.HttpResponse(json_string, status_code=200)
        # TEST:
        # result = {
        #         "class_label": 'test',
        #         "class_probability": 0.98
        #     }
        # # convert (serialize) dictionary to JSON string object
        # json_string = json.dumps(result)
        # return func.HttpResponse(json_string, status_code=200)
    else:
        return func.HttpResponse("Error: Unknown img url",status_code=400)