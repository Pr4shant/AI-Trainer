import cv2
import mediapipe as mp
import time
import math
import numpy as np

from numpy.lib.function_base import angle


class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detection_Confidence=0.5, tracking_Confidence=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detection_Confidence
        self.trackCon = tracking_Confidence

        self.markers_list = []
        self.positional_markers = []
        self.positional_angles = []
        self.rep_count = 0
        self.direction = 0

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        angle = 0
        if len(self.lmList)!= 0:
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                math.atan2(y1 - y2, x1 - x2))
            if angle < 0:
                angle = -1*angle
            if angle > 180:
                angle = 360 - angle
            
        # Draw
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 2)
                cv2.circle(img, (x1, y1), 5, (0, 0, 0), cv2.FILLED)
                #cv2.circle(img, (x1, y1), 7, (255, 255, 255), 2)
                cv2.circle(img, (x2, y2), 5, (0, 0, 0), cv2.FILLED)
                #cv2.circle(img, (x2, y2), 7, (255, 255, 255), 2)
                cv2.circle(img, (x3, y3), 5, (0, 0, 0), cv2.FILLED)
                #cv2.circle(img, (x3, y3), 7, (255, 255, 255), 2)
                cv2.putText(img, str(int(angle)), (x2 + 30, y2),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        return angle

    def generate_position_markers(self, img, name = ""):
        if name == "pushup":
            # Right side
            self.positional_markers = [16, 14, 12, 24, 26, 28]
            self.positional_angles = [int(self.findAngle(img, 16,14,12)),
                                      int(self.findAngle(img, 14,12,24)),
                                      int(self.findAngle(img, 12,24,26)),
                                      int(self.findAngle(img, 24,26,28)),]

        if name == "plank":
            #Left Right side
            self.positional_markers = [16, 14, 12, 24, 26, 28]
            self.positional_angles = [int(self.findAngle(img, 16,14,12)),
                                      int(self.findAngle(img, 14,12,24)),
                                      int(self.findAngle(img, 12,24,26)),
                                      int(self.findAngle(img, 24,26,28)),]

    def generate_markers_reps(self, name = ""):
        if name == "pushup":
            # Tracking right arm
            self.markers_list = [12, 14, 16]
        
    def reps(self, img, name = "", draw = True):
        self.generate_markers_reps(name)
        p1, p2, p3 = self.markers_list
        angle = self.findAngle(img, p1, p2, p3)
        self.percentage = np.interp(angle, (70, 160), (100,0))
        bar = np.interp(angle, (70, 160), (650, 100))

        if self.percentage == 100:
            if self.direction == 0:
                self.rep_count += 0.5
                self.direction =1
        if self.percentage == 0:
            if self.direction == 1:
                self.rep_count += 0.5
                self.direction = 0

        if draw:
            #cv2.rectangle(img, (450, 100), (350, 100), (0,0,0), 3)
            #cv2.rectangle(img, (450, int(bar)), (350, 100), (0,0,0), cv2.FILLED)
            cv2.putText(img, f'{int(self.percentage)} %', (450, 75), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0,0,0), 2)
        cv2.putText(img, f'Reps:{str(int(self.rep_count))}', (45, 170), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)
        
        return self.rep_count

    def initial_position(self, img, name = ""):
        if name == "pushup":
            ideal_position_angles = [165, 60, 170, 165]
            for i in range(len(self.positional_angles)):
                if (self.positional_angles[i] > ideal_position_angles[i]+10) or (
                    self.positional_angles[i] < ideal_position_angles[i]-10):
                    #print(self.positional_angles[i])
                    return False
            
        if name == "plank":
            ideal_position_angles = [65, 65, 170, 170]
            for i in range(len(self.positional_angles)):
                if (self.positional_angles[i] > ideal_position_angles[i]+10) or (
                    self.positional_angles[i] < ideal_position_angles[i]-10):
                    #print(self.positional_angles[i])
                    return False
        return True


def main():
    #cap = cv2.VideoCapture('test2.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        #success, img = cap.read()
        img = cv2.imread('pushup.jpg')
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()