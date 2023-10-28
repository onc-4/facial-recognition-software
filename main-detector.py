import numpy as np
import cv2 as cv
import os
import time

faceCascade = cv.CascadeClassifier("Haar Cascades/haarcascade_frontalface_default.xml")
eyeCascade = cv.CascadeClassifier("Haar Cascades/haarcascade_eye.xml")
cap = cv.VideoCapture(0) # initializing video capture
cap.set(3, 640)
cap.set(4, 480)

face_id = input("\nEnter user id number: ")
print("Capturing facial scan, look straight at the camera and hold still")

num_facial_scans = 0


while (True):
    ret, colour = cap.read()
    grey = cv.cvtColor(colour, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        grey, 
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv.rectangle(colour,(x,y),(x+w,y+h),(255,0,0),2)
        features_grey = grey[y: y+h, x: x+w]
        features_colour = colour[y: y+h, x: x+w]

        eyes_detection = eyeCascade.detectMultiScale(
            features_grey,
            scaleFactor=1.5,
            minNeighbors=10,
            minSize=(5, 5)  
        )

        for (ex, ey, ew, eh) in eyes_detection:
            cv.rectangle(features_colour, (ex, ey), (ex + ew, ey+ eh), (0, 225, 0), 2)

        num_facial_scans += 1
        cv.imshow('frame', colour)
        time.sleep(0)
        cv.imwrite("dataset/User." + str(face_id) + "." + str(num_facial_scans) + ".jpg", grey[y: y+h, x: x+w])



    esc = cv.waitKey(30) & 0xff
    if esc == 27:
        break
    
    if num_facial_scans >= 60:
        break
        
print("Scans complete")
cap.release()
cv.destroyAllWindows()