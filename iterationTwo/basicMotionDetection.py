from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils.video import VideoStream
from imutils import paths
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
    help="path to the input image")
ap.add_argument("-w", "--win-stride", type=str, default="(8, 8)",
    help="window stride")
ap.add_argument("-p", "--padding", type=str, default="(16, 16)",
    help="object padding")
ap.add_argument("-s", "--scale", type=float, default=1.05,
    help="image pyramid scale")
ap.add_argument("-m", "--mean-shift", type=int, default=-1,
    help="whether or not mean shift grouping should be used")
args = vars(ap.parse_args())
args = vars(ap.parse_args())

#assign command-line arguments to variables
winStride = eval(args["win_stride"])
padding = eval(args["padding"])
meanShift = True if args["mean_shift"] > 0 else False
 
# if the video argument is None, then we are reading from camera
if args.get("video", None) is None:
    vs = VideoStream(src=0).start() #Going to change souece to try and
                                    #capture from the thermal camera
    time.sleep(2.0)

# otherwise read from a video file
else:
    vs = cv2.VideoCapture(args["video"])
 
#initialize the first frame in the video stream
firstFrame = None

#create a person detecor object
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while(True):
    
    #Let's get fram data
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    frame = imutils.resize(frame, width=400)
    
    #detect in frame... hopefully real-time
    (rects, weights) = hog.detectMultiScale(frame, winStride=winStride, padding=padding, scale=args["scale"], useMeanshiftGrouping=meanShift)
    
    #bounding boxes and occupancy number
    numPeople = 0;
    for (x, y, w, h) in rects:
        numPeople+=1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = numPeople
    
    #quick test
    cv2.putText(frame, "Num Occupants: {}".format(text), (10, 20),
    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Occupants", frame)
    key = cv2.waitKey(1) & 0xFF

