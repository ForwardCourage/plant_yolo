import json
from flask import Flask, Response, jsonify
from ultralytics import YOLO
import os 
from PIL import Image
import numpy as np



if os.path.exists('best.pt'):
    model = YOLO('best.pt')

else:
    model = YOLO('yolov8n.pt')


def load_json(input):
    img_array = json.load()



def model_predict(input):
    results = model(input)
    return results

def export_json(input):
    results = model_predict(input)
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        
    im.save('results.jpg')  # save image

    im_arr = np.array(Image.open('results.jpg'))
    im_arr = im_arr.astype(np.float32)
    output = {'Array':im_arr}
    return jsonify(output)
