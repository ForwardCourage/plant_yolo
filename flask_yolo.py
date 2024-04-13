import cv2
import onnxruntime as ort
import os
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np



from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

classes = [
    '紫花藿香薊','紫花藿香薊花','紫花藿香薊葉',
    '刺莧','刺莧花','刺莧葉',
    '大花咸豐草','大花咸豐草花','大花咸豐草葉','大花咸豐草種子',
    '巴拉草','巴拉草種子',
    '落地生根','落地生根葉',
    '雞冠花','雞冠花花', '雞冠花葉',
    '孟仁草', '孟仁草花',
    '假蓬草', '假蓬草花', '假蓬草葉',
    '昭和草','昭和草花', '昭和草葉',
    '狗牙根',
    '牛筋草', '牛筋草花',
    '粗毛小米菊',
    '馬纓丹','馬纓丹花','馬纓丹葉',
    '銀合歡','銀合歡花','銀合歡葉','銀合歡種子',
    '大黍','大黍種子',
    '小花蔓澤蘭', '小花蔓澤蘭花','小花蔓澤蘭葉',
    '象草','象草花',
    '紅毛草','紅毛草花',
    '芒草', '芒草花',
    '合果芋',
    '王爺葵','王爺葵花', '王爺葵_葉',
]

# Load the YOLOv8 model
model = YOLO('best.pt')

for i in range(len(classes)):
    model.names[i] = classes[i]



cap = cv2.VideoCapture(0)

model = YOLO('yolov8n.pt')


# def resize_img_2_bytes(image, resize_factor, quality):
#     bytes_io = BytesIO()
#     img = Image.fromarray(image)

#     w, h = img.size
#     img.thumbnail((int(w * resize_factor), int(h * resize_factor)))
#     img.save(bytes_io, 'jpeg', quality=quality)

#     return bytes_io.getvalue()


# def get_image_bytes():
#     success, img = cap.read()
#     if success:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_bytes = resize_img_2_bytes(img, resize_factor=1.0, quality=60)
#         return img_bytes

#     return None

# def gen_frames():
#     while True:
#         img_bytes = get_image_bytes()
#         if img_bytes:
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

@app.route("/" ,methods=['POST', 'GET'])
def index():
    if request.method=='POST':
        print(request.json())
    return 'Hello World!'

# @app.route('/api/stream')
# def video_stream():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


    
app.run(debug=True)