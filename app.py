from flask import Flask, Response, render_template
import random
import cv2
import numpy as np
from flask_socketio import SocketIO, emit
from threading import Thread, Event
import time

DEBUG_SHOW_SPOT=True #True #False #True
# 0 == cam 2 (capture device #1)
# 1 == localhost stream port 2727
CAM2_MODE = 0
app = Flask(__name__)
socketio = SocketIO(app)


# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
count =0
countdi={"c": 0, "i": None}
yyy=410

def aplis_predicate(x, y):
    return (x-500)**2/420.5**2+(y-yyy)**2/122.5**2

def applis(x, y):
    return aplis_predicate(x, y) <=1

def applis_l(x, y):
    return aplis_predicate(x, y) ==1


def ieks_rinka(xx, xy, xw, xh):
    horp = 400
    verp = 500
    if(xy <horp): #_|
        if(xx+xw < verp): #<-
            x=xx+xw
            y=xy
        else: 
            x=xx
            y=xy
    else: #-|
        if(xx+xw < verp): #<-
            x=xx+xw
            y=xy+xh
        else: 
            x=xx
            y=xy+xh
    return applis(x, y)

def gen_frames():  
    global count, countdi
    print("framed")
    cap = cv2.VideoCapture(1) if CAM2_MODE == 0 else \
    cv2.VideoCapture("http://jtag.me:2727/") # set the camera, 0 for default
    while  not thread_stop_event2.isSet():
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                   center_x = int(detection[0] * width)
                   center_y = int(detection[1] * height)
                   w = int(detection[2] * width)
                   h = int(detection[3] * height)
                   x = int(center_x - w / 2)
                   y = int(center_y - h / 2)
                   if(ieks_rinka(x, y, w, h)):
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                        count=count+1

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indexes) > 0 and isinstance(indexes, tuple):
            indexes = indexes[0]
        count =0
        for i in indexes:
            i = i[0] if isinstance(i, tuple) else i
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "car":
                count = count+1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
                
        if(DEBUG_SHOW_SPOT):
            alpha = 0.75
            color=(125, 25, 125)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            y, x = np.ogrid[:height, :width]
            mask = applis(x, y)
            overlay = np.zeros_like(frame, dtype=np.uint8)
            overlay[:, :] = color + (0,)      
            overlay[mask] = color + (int(alpha * 255),)
            result = frame.copy()
            frame[mask] = cv2.addWeighted(result[mask], 1 - alpha, overlay[mask], alpha, 0)

            mask = applis_l(x, y)
            frame[mask] = [125, 25, 125, 77]


        ret, buffer = cv2.imencode('.png', frame)
        socketio.emit('number', {'data': (bytearray([count]) + buffer.tobytes()).hex()})

@app.route('/')
def index():
    global thread2
    if not thread2.is_alive():
        thread2 = Thread(target=gen_frames)
        thread2.start()
    return render_template('index.html')


thread2 = Thread(target=gen_frames)
thread_stop_event2 = Event()

@socketio.on('connect')
def handle_connect():
    global thread2
    if not thread2.is_alive():
        thread2 = Thread(target=gen_frames)
        thread2.start()

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
    #thread_stop_event2.set()
