import cv2 as cv
import numpy as np
import os
from PIL import Image

path = "dataset"
face_rec = cv.face.LBPHFaceRecognizer.create()
face_dec = cv.CascadeClassifier("Haar Cascades/haarcascade_frontalface_default.xml")

def is_image_file(filename): # makes sure file is a image
    return any(filename.endswith(extension) for extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp'])

def getImageWithLabels(path):
    image_path = [os.path.join(path, f) for f in os.listdir(path) if is_image_file(f)]
    faceScans = []
    Ids = []

    for paths in image_path:
        PIL_image = Image.open(paths).convert('L')
        image_np = np.array(PIL_image, "uint8")
        id = int(os.path.split(paths)[-1].split(".")[1])
        faces = face_dec.detectMultiScale(image_np)

        for (x,y,w,h) in faces:
            faceScans.append(image_np[y: y+h, x: x+w])
            Ids.append(id)

    return faceScans, Ids

faceScans, Ids = getImageWithLabels(path)
face_rec.train(faceScans, np.array(Ids))

face_rec.write("facial rec trainer/trainer.yml") 
print("\n {0} faces trained".format(len(np.unique(Ids))))
