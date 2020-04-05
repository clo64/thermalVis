import numpy as np
import cv2
import time
import imutils
import time
import requests
import os

def thermalDetection(wPipe):
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
    writeModulo = 0;
    while True:
        #test changing to grab
        
        _, frame = cap.read()
        # if frameModulo%20 == 0:
        #frame_id += 1
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
        #print(len(boxes))

        for i in range(len(boxes)):
            if i in indexes:     
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                print(label)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(frame, label, (x, y -10), font, .2, (0, 255, 0), 3)
    
        frame = imutils.resize(frame, width=400)
        
        if(writeModulo%10==0):
            os.write(wPipe, str.encode(str(len(boxes))))
            
        writeModulo += 1
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
def httpPost(rPipe):
    while True:
        #time.sleep(8)
        readIt = os.read(rPipe, 10)
        readIt = readIt.decode()
        print("readIt")
        print(readIt)
        print("readIt")
        pload = {
            "Room_Number": "1",
            "Floor": "1",
            "Thermal_Occupants": readIt,
            "All": "0"
            }
        postIt = requests.post('http://occupancy-detection.herokuapp.com/api/thermaldata', json=pload)
        print("Sending Data")
        print(postIt.text)
    #print("reading")
    #while True:
    #     readIt = os.read(rPipe, 20)
    #     print(readIt.decode())
      
if __name__ == '__main__':
    
    #create pip for IPC occupancy values
    rPipe, wPipe = os.pipe()
    
    #fork processes, one to for thermal detection
    #one for making HTTP posts with our data
    pid = os.fork()
    
    if(pid):
        thermalDetection(wPipe)
    else:
        httpPost(rPipe)
#End OpenCV Process