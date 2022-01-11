import cv2
import time
import mediapipe as mp


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
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 7, (255,0,255), cv2.FILLED)
        
        return lmList
        


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
            print(lmList[14])
            cv2.circle(img, (lmList[14][1],lmList[14][2]),5, (0,0,255), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__" :
    main()