import cv2
import numpy as np
vx =0

cap = cv2.VideoCapture(0)
range = [10000,20000]
pid = [0.4,0.4,0]
pError = 0
def find_face(img):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img_gray,1.3,5)

    face_c=[]
    face_area=[]
    for (x,y,w,h) in faces:
        cv2.rectangle(img , (x,y),(x+w,y+h),(0,255,0),2)
        cx = x+w //2
        cy = y+h //2
        area = w*h
        face_c.append([cx,cy])
        face_area.append(area)
    if len(face_area) !=0:
        i = face_area.index(max(face_area))
        return img ,[face_c[i],face_area[i]]
    else:
        return img , [[0,0],0]
def tracking(info,w,pid,pError):
    area=info[1]
    x , y  =info[0]
    vx =0
    error = x -w//2
    yaw = pid[0]*error + pid[1]*(error -pError)
    yaw = int(np.clip(yaw,-100,100))


    if area>range[0] and area<range[1]:
        vx =0 
    elif area >range[1]:
        vx = -2
    elif area < range[0] and area !=0:
        vx = 2
    elif x == 0:
        yaw = 0
        error =0
    print(vx , yaw)
    return error
    
    
while True:
    _,img =cap.read()
    img , info = find_face(img)
    w = img.shape[1]
    pError = tracking(info ,w,pid,pError)
    cv2.imshow("d",img)
    cv2.waitKey(1)
