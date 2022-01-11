import cv2
import time
import mediapipe as mp
import PoseEstimationModule as pem
import numpy as np

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
cTime = 0

detector = pem.PoseDetector(detectionConf=0.75)

count = 0
dir = 0

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.getPosition(img, draw=False)
    if len(lmList)!=0:
        angle = detector.findAngle(img, 16, 14, 12)
        per = np.interp(angle, (20,170), (0,100))
        #print(per)

        #check for the curls
        if per == 100:
            if dir == 0:
                count +=0.5
                dir = 1

        if per == 0:
            if dir == 1:
                count +=0.5
                dir = 0 
        
        print(count)
        cv2.rectangle(img, (0,458),(250, 728), (0,255,0), cv2.FILLED)
        cv2.putText(img, f'{int(count)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cv2.imshow("Image", img)
    k = cv2.waitKey(1)

    if k==27:
        break