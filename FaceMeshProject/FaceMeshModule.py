import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, mode=False, numFaces = 2, isRefine = False,minDetConf = 0.5, minTrackConf=0.5) -> None:
        self.mode = mode
        self.numFaces = numFaces
        self.isRefine = isRefine
        self.minDetConf = minDetConf
        self.minTrackConf = minTrackConf

        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.numFaces, self.isRefine,self.minDetConf,self.minTrackConf)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius = 2)


    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []        

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, None, self.drawSpec)
                
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih), 
                    face.append([id,x,y])

                faces.append(face)
        return img, faces
    


def main():
    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0
    faces = []
    detector = FaceMeshDetector()

    while(True):
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)

        if len(faces)!=0:
            print(faces[0])
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        k = cv2.waitKey(1)

        if(k == 27):
            break

if __name__ == '__main__':
    main()