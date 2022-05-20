import cv2
import numpy as np
import time
import os
import module.HandTrackingModule as htm

##################
brushThickness = 10
eraseThickness = 50
##################


folderPath = "designs"
myList = os.listdir(folderPath)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]  # default header style
drawColor = (255, 0, 255)

# Capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.7)  # high detection confidence

xp, yp = 0, 0  # start of line

# drawing canvas
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)  # flip the image to draw easily

    # 2. Find Hand Landmarks
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)
        # tip of middle finger to draw
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up, just one finger can draw
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If Selection mode  -- Two fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0   # reset
            # print("Selection Mode")
            # change selection click
            if y1 < 180:  # finger in the header zone
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)  # erase
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If  Drawing Mode  -- Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)
            # print("Drawing Mode")
            # Draw lines
            if xp == 0 and yp == 0:  # first frame, draw a point
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):  # erase
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraseThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraseThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            # setting new start of line
            xp, yp = x1, y1

    # creating two inverse images  two gray images
    imgCray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgCray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header  # setting the head image
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)   # combine two images
    cv2.imshow("AI Virtual Painter", img)
    # cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
