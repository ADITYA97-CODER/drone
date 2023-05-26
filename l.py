import cv2
import numpy as np
model = cv2.face.LBPHFaceRecognizer_create()
cap = cv2.VideoCapture(0)
model.read('face-trainer.yml')
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)


    return faces
while True:
    s , right= cap.read()
    facesR = face_detector(right)
    if len(facesR)>=1:
     for(x,y,w,h) in facesR:
        roi = right[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
        face = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
        if confidence>50:
         cx = x+w //2
         cy = y+h //2
         width = right.shape[1]
         area = w*h
         info = [[cx,cy],area]
         cv2.rectangle(right, (x,y),(x+w,y+h),(0,255,0),2)
         cv2.putText(right, "Aditya", (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
         percent="Matched - "+str(confidence)+" %"
         cv2.putText(right, percent, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
         cv2.imshow('Face CropperR', right)
        else:
            cv2.putText(right, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face CropperR', right)
    else:
        cv2.putText(right, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face CropperR', right) 
    cv2.waitKey(1)
