# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:32:12 2020

@author: Admin
"""

import cv2
import numpy as np

class_names = open("D:\\Python Projects\\OpenCV\\Object Detection - YOLO\\coco.names").read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet("D:\\Python Projects\\OpenCV\\Object Detection - YOLO\\yolov3.cfg", "D:\\Python Projects\\OpenCV\\Object Detection - YOLO\\yolov3.weights")

layer_names = net.getLayerNames()
layerOutputs = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0,255,size=(len(class_names),3))
# Prepare labels of the network (20 class labels + background):
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), swapRB=True, crop=False)
    
    height,width,channels = frame.shape
    net.setInput(blob)
    detections = net.forward(layerOutputs)
    
    boxes = []
    confidences = []
    class_ids = []
     
    for output in detections:
    # loop over each of the detections
       for detection in output:
        # Get class ID and confidence of the current detection:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.25:
            # Scale the bounding box coordinates (center, width, height) using the dimensions of the original image:
            center_x= int(detection[0]*width)
            center_y= int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            # Calculate the top-left corner of the bounding box:
            x = int(center_x - (width / 2))
            y = int(center_y - (height / 2))

            # Update the information we have for each detection:
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(class_names[class_ids[i]])
            confidence= confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
            

    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
         break
    
cap.release() 
cv2.destroyAllWindows() 