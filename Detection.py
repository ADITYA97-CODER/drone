import cv2
import numpy as np
from picamera2 import Picamera2
# Grab images as numpy arrays and leave everything else to OpenCV.


picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
from os import listdir
from os.path import isfile, join
import time
import serial
from pymavlink import mavutil
import argparse
import sys
import time
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
the_connection = mavutil.mavlink_connection('/dev/ttyS0', baud=57600)

the_connection.mav.command_long_send(the_connection.target_system,the_connection.target_component,mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                      0,1,0,0,0,0,0,0)
def run(model: str, num_threads: int,
        enable_edgetpu: bool):
  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)
  return detector

    
detector = run('efficientdet_lite0.tflite',4,False)

range = [10000,20000]
pid = [0.4,0.4,0]
pError = 0


model = cv2.face.LBPHFaceRecognizer_create()

model.read('face-trainer.yml')




face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)


    return faces
def tracking(info,w,pid,pError):
    area=info[1]
    print(area)
    x , y  =info[0]
    vx =0
    error = x -w//2
    yaw = pid[0]*error + pid[1]*(error -pError)
    yaw = int(np.clip(yaw,-100,100))


    if area>range[0] and area<range[1]:
        vx =0 
    elif area >range[1]:
        vx = -0.5
    elif area < range[0] and area !=0:
        vx = 0.5
    elif x == 0:
        yaw = 0
        error =0
    if yaw>0:
       print(yaw)
       the_connection.mav.command_long_send(the_connection.target_system,the_connection.target_component,mavutil.mavlink.MAV_CMD_CONDITION_YAW,0,yaw,0,1,1,0,0,0)
    else:
        the_connection.mav.command_long_send(the_connection.target_system,the_connection.target_component,mavutil.mavlink.MAV_CMD_CONDITION_YAW,0,(-1*yaw),0,-1,1,0,0,0)
    the_connection.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(10,the_connection.target_system,the_connection.target_component,mavutil.mavlink.MAV_FRAME_BODY_NED,int(0b110111000111),0,0,0,vx,0,0,0,0,0,0,0))
    return error

   



while True:
    im = picam2.capture_array()
    
    right= im[:,0:320]
    left = im[:,320:640]
    """ret, image = cap.read()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = vision.TensorImage.create_from_array(rgb_image)
    detection_result = detector.detect(input_tensor)
    im  = utils.visualize(rgb_image, detection_result)
    cv2.imshow('object_detection',im)"""
    
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
         pError = tracking(info,width,pid,pError)
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
    facesL = face_detector(left)
    if len(facesL)>=1:
     for(x,y,w,h) in facesL:
        roi = left[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
        face = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
        if confidence>50:
         cx = x+w //2
         cy = y+h //2
         width = left.shape[1]
         area = w*h
         info = [[cx,cy],area]
         pError = tracking(info,width,pid,pError)
         cv2.rectangle(left, (x,y),(x+w,y+h),(0,255,0),2)
         cv2.putText(left, "Aditya", (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
         percent="Matched - "+str(confidence)+" %"
         cv2.putText(left, percent, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
         cv2.imshow('Face CropperL', left)
        else:
            cv2.putText(left, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face CropperL', left)
    else:
        cv2.putText(left , "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face CropperL', left)
         


    


    if cv2.waitKey(1)==ord("s"):
        break


cap.release()
cv2.destroyAllWindows()
