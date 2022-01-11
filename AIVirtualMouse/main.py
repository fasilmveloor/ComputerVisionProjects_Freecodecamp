#1 . Import Image and find hand landmarks
#2 . Get the tip of the index and middle finger
#3 . Check Fingers are up
#4 . Only Index Finger:Moving Mode
#5 . Convert coordinates
#6 . Smoothen values
#7 . Move Mouse
#8 . Both index and middle finger are up
#9 . Find distances between fingers
#10. Click mouse if distance short



import cv2
import time
import mediapipe as mp
import HandTrackingModule as htm
import numpy as np
import imutils
import autopy

wCam, hCam = 1080, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
cTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

detector = htm.HandDetector(detectionCon=0.75)

count = 0
dir = 0


wScr , hScr = autopy.screen.size()
frameR = 100
smoothening = 5

imgCanvas = np.zeros((810, 1080, 3), np.uint8)

while True:
    
    success, img = cap.read()
    img = imutils.resize(img, width=1080)
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        #Tip of index finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        cv2.reactangle(img, (frameR, frameR), (wCam -frameR, hCam-frameR), (255,0,255),2)

        if fingers[1] and fingers[2]==0 and fingers.count(1)==1:
            
            x3 = np.interp(x1, (frameR,wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR,hCam-frameR), (0,hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            
            autopy.mouse.move(wScr-x3, y3)
            cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
            plocX , plocY = clocX, clocY

        if fingers[1] and fingers[2] and fingers.count(1)==2:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0,255,0), cv2.FILLED)
                autopy.mouse.click()

    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cv2.imshow("Image",img) 
    k = cv2.waitKey(1)

    if k==27:
        break