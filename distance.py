import cv2
from math import sqrt , pow
import time
import torch
import argparse
velocity =5
cap =cv2.VideoCapture('vf.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
from pymavlink import mavutil
the_connection = mavutil.mavlink_connection('udpin:127.0.0.1:14550')
the_connection.wait_heartbeat()
a =[]
n =0
x = 0
po=[208,208]
y =0
prev_po =[208,208]
area_previous = 0 
area_current = 0
def PID_X(x1,y1,po):
  propy= y1-po[1]
  diffy = propy-prev_po[1]
  propx = x1-po[0]
  diffx = (propx-prev_po[0])
  kp= 0.4
  kd =0.4
  total_movementy = int((kp*propy)+(kd*diffy))
  total_movementx =int((kp*propx) + (kd*diffx))
  prev_po[0]= propx
  prev_po[1]=propy
  return total_movementx , total_movementy
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
classes= model.names
def score_frame(frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        frame = [frame]
        results = model(frame)
        #print(results.pandas().xyxy)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        
        return labels, cord
def plot_boxes(results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if int(labels[i])==0:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                l = int(((x1+x2)/2))
                m = int(((y1+y2)/2))
                
                p =frame[l][m]
            
                
                bgr = (0,0,255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.circle(frame , (l,m),1,(255,15,0),1)
                cv2.putText(frame, classes[int(labels[i])]+str(float(row[4])), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                break
                
                
        
        
        return frame
    

while True:
    n=n+1
    success , frame = cap.read()
    frame = cv2.resize(frame,(416,416))
    result = score_frame(frame)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    frame= plot_boxes(result , frame)
    labels , cord = result
    l= len(labels)
    for i in range(l):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                if area_previous==0:
                     area_previous= (x2-x1)*(y2-y1)

                area_current = (x2-x1)*(y2-y1)
                if area_current!=area_previous:
                    distance = velocity*0.001 *((area_current)/(area_current-area_previous))
                    area_previous = area_current
                    #frame = cv2.putText(frame , str(distance),((x1,y1)),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                l = int(((x1+x2)/2))
                m = int(((y1+y2)/2))
                px ,py = PID_X(l,m,po) 
                po[0]= int(po[0]+px)
                po[1]= int(po[1]+py)
                print(px)
                if px>0:
                    the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,
                                     mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, px, 0, 1, 1, 0, 0, 0)
                    the_connection.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(10, the_connection.target_system,
                        the_connection.target_component, mavutil.mavlink.MAV_FRAME_BODY_NED, int(0b110111000111), 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0))

                else :
                    the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,
                                     mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0, (-1*px),0, -1, 1, 0, 0, 0)
                break    
    frame = cv2.circle(frame,(po[0],po[1]),4,(0,0,255),5)
    cv2.imshow("dasd",frame)
    cv2.waitKey(1)        



     
        