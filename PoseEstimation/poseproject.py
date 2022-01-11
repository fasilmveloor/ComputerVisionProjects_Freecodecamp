import cv2
import time
import mediapipe as mp
import PoseEstimationModule as pm


pTime = 0
cTime = 0

    #cap = cv2.VideoCapture('PoseVideos/1.mp4')
cap = cv2.VideoCapture(0)
detector = pm.PoseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.getPosition(img)
    if len(lmList)!=0:
        print(lmList[14])
        cv2.circle(img, (lmList[14][1],lmList[14][2]),5, (0,0,255), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
