import numpy as np
import cv2
import time
import imutils
import multiprocessing
import requests

#Goal is to send the number of occupants
#from the OpenCV process to our looping


# Load Yolo
net = cv2.dnn.readNet("yolov3Chuck_final.weights", "yolov3Chuck.cfg")
classes = ['person_up']
#with open("obj.names", "r") as f:
#    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading camera
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
starting_time = time.time()
frame_id = 0
sendModulo = 0;

while True:
    _, frame = cap.read()
    frame_id += 1
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
            
                #rectangle coords
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
            
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
            
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    #number_object_detected = len(boxes)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)

    for i in range(len(boxes)):
        if i in indexes:     
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            print(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(frame, label, (x, y -10), font, .2, (0, 255, 0), 3)
    
    frame = imutils.resize(frame, width=400)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    
    if(sendModulo%20 == 0):
        pload = {
            "RoomNumber": "1",
            "Floor": "1",
            "NumberOccupants": str(len(boxes)),
            "All": "0",
            "CreateNew": "0"
            }
        postIt = requests.post('http://occupancy-detection.herokuapp.com/api/thermaldata', json=pload)
        print(postIt.text)
        print("DB Updated")
    sendModulo += 1
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
#End OpenCV Process
    
#of objects detected            
#print(len(boxes))
