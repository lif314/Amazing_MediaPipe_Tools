import cv2
import numpy as np
import time
import os
import module.HandTrackingModule as htm

folderPath = "designs"
myList = os.listdir(folderPath)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]  # default header style

# Capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

