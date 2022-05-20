import time
import numpy as np
import module.HandTrackingModule as htm
import math
import cv2
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume # change volume


################################
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
################################

# volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

# hand detector
detector = htm.handDetector(detectionCon=0.7)  # 提高检测阈值
pTime = 0  # previous time  cTime: current time
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    # hand landmark list
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2]   # 大拇指
        x2, y2 = lmList[8][1], lmList[8][2]   # 食指
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 两者的中心点
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)  # 计算两个指点之间的距离 平方和
        # print(length)
        # Hand range 50 - 300
        # Volume Range -65 - 0
        vol = np.interp(length, [50, 300], [minVol, maxVol])  # 线性插值
        volBar = np.interp(length, [50, 300], [400, 150])  # volume bar
        volPer = np.interp(length, [50, 300], [0, 100])    # volume percentage
        # print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)   # change volume
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
    # show the volume var
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    # calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)
    cv2.imshow("Img", img)
    cv2.waitKey(1)
