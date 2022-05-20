import cv2
import time
import os
import module.HandTrackingModule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Finger Images
folderPath = "fingers"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))


pTime = 0
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]  # point to detect open gesture  每个手指的指尖
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    # TODO 如何设计算法检测手指的语义
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers 正手检测是否打开手指：指尖的纵坐标大小
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)

        if totalFingers - 1 >= 0:
            h, w, c = overlayList[totalFingers - 1].shape
            img[0:h, 0:w] = overlayList[totalFingers - 1]

        # cv2.rectangle(img, (20, 225), (170, 450), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (30, 450), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)