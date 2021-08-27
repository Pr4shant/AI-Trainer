from math import trunc
from typing import Tuple
import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("situp.mp4")
detector = pm.poseDetector(detection_Confidence=0.7, tracking_Confidence= 0.7)
pTime = 0
draw = True
initial_position = False

while True:
   success, img = cap.read()
    
   #img = cv2.imread("plank2.jpg")
   img = cv2.resize(img, (1080, 480))
   
   #img = cv2.flip(img, 1)

   img = detector.findPose(img, False)
   lmList = detector.findPosition(img, False)

   #Display frame rate
   cTime = time.time()
   fps = 1 / (cTime - pTime)
   pTime = cTime
   cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
               (255, 0, 0), 2)

   
   detector.generate_position_markers(img, "pushup")

   if initial_position!=True:
      initial_position = detector.initial_position(img, "plank")
         
   if initial_position == True:
      cv2.rectangle(img,(10,60), (30,80), (0,255,0), cv2.FILLED)
      #detector.reps(img, "pushup")

   cv2.imshow("Image", img)
   cv2.waitKey(1)