import cv2 as cv
import numpy as np
import os

face_rec = cv.face.LBPHFaceRecognizer.create()
face_rec.read("facial rec trainer/trainer.yml")

face_cascade = cv.CascadeClassifier("Haar Cascades/haarcascade_frontalface_default.xml")
text_font = cv.FONT_HERSHEY_COMPLEX

id = 0

names_by_id = [None, "Omer", "Hamzah"]
live_cam = cv.VideoCapture(0)
live_cam.set(3, 640)
live_cam.set(4, 480)

min_face_h = 0.1*live_cam.get(3)
min_face_w = 0.1*live_cam.get(4)

while True:
    ret, img = live_cam.read()
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        grey,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(int(min_face_w), int(min_face_h))
    )
    
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), (0,225,0), 2)
        id, confidence_of_match = face_rec.predict(grey[y: y+h, x: x+w])

        if (confidence_of_match < 100):
            id = names_by_id[id]
            confidence_of_match = "  {0}%".format(round(confidence_of_match))
            
        
        else:
            id = 'unknown'
            confidence_of_match = "  {0}%".format(round(0))
            

        cv.putText(img, str(id), (x+5,y-5), text_font, 1, (255,255,255), 2)
        cv.putText(img, str(confidence_of_match), (x+5,y+h-5), text_font, 1, (255, 255, 0), 1)

    cv.imshow("live", img)

    k = cv.waitKey(10) & 0xff 
    if k == 27:
        break

live_cam.release()
cv.destroyAllWindows()