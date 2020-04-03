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
    frameModulo = 0;
    while True:
        #test changing to grab
        # _, frame = cap.read()
        ret = cap.grab()
        if frameModulo%10 == 0:
            print("hi")
            
        frameModulo = frameModulo+1
        
    cap.release()
    cv2.destroyAllWindows()
    
def httpPost(rPipe):
    while True:
        time.sleep(2)
        readIt = os.read(rPipe, 1)
        readIt = readIt.decode()
        print("readIt")
        print(readIt)
        print("readIt")
        pload = {
            "RoomNumber": "1",
            "Floor": "1",
            "NumberOccupants": readIt,
            "All": "0",
            "CreateNew": "0"
            }
        postIt = requests.post('http://occupancy-detection.herokuapp.com/api/thermaldata', json=pload)
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