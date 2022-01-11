import cv2
import time
import mediapipe as mp
import math

class PoseDetector():
    def __init__(self, mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False,
                    smooth_segmentation=True, detectionConf=0.5, trackConf=0.5) -> None:
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionConf = detectionConf 
        self.trackConf = trackConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation, 
                                        self.smooth_segmentation, self.detectionConf, self.trackConf)

    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
        
    def getPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 7, (255,0,255), cv2.FILLED)
        
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        #getting landmark
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        #calculating angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle = angle + 360
        #print(angle)

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1,y1), 7, (255,0,0), cv2.FILLED)
            cv2.circle(img, (x1,y1), 12, (255,0,0), 2)
            cv2.circle(img, (x2,y2), 7, (255,0,0), cv2.FILLED)
            cv2.circle(img, (x1,y1), 12, (255,0,0), 2)
            cv2.circle(img, (x3,y3), 7, (255,0,0), cv2.FILLED)
            cv2.circle(img, (x1,y1), 12, (255,0,0), 2)
            #cv2.putText(img, str(int(angle)), (x2-50, y2+20), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

        return angle


def main():
    pTime = 0
    cTime = 0

    #cap = cv2.VideoCapture('PoseVideos/1.mp4')
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPosition(img)
        if len(lmList)!=0:
            detector.findAngle(img, 16, 14, 12)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        k = cv2.waitKey(1)
        if k==27:
            break


if __name__ == "__main__" :
    main()