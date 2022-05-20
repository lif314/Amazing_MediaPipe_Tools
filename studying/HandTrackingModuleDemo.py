import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self,
                 mode=False,
                 maxHandsNum=2,
                 minDetectionConfidence=0.5,
                 minTrackingConfidence=0.5):
        # parameter
        self.mode=mode
        self.maxHandsNum=maxHandsNum
        self.minDetectionConfidence=minDetectionConfidence
        self.minTrackingConfidence=minTrackingConfidence

        # Hand detection model
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
                       self.mode,      # treat the input images
                       self.maxHandsNum,
                       self.minDetectionConfidence,  # 低于该值进行重新检测
                       self.minTrackingConfidence)

        self.mpDraw = mp.solutions.drawing_utils  # draw results

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # Processes an RGB image
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:    # 没有检测到数据时为None
            for handLms in self.results.multi_hand_landmarks:  # single hand landmark
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)  # connections
        return img

    def findPosition(self,img, handId=0, draw=True): # handId 21
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handId]
            for id, landmark in enumerate(myHand.landmark):  # finger id andlandmark
                height, width, c = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                lmList.append([id, cx, cy])
                if draw:  # find a point location
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmList

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        # 获取绝对坐标
        lmList = detector.findPosition(img)
        # if len(lmList) != 0:
        #     print(lmList[4])

        # fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, "FPS:" + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()