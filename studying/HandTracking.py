import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# Hand detection model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
               static_image_mode=False,      # treat the input images
               max_num_hands=2,
               min_detection_confidence=0.5,  # 低于该值进行重新检测
               min_tracking_confidence=0.5)

mpDraw = mp.solutions.drawing_utils  # draw results


pTime = 0
cTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)  # Processes an RGB image
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:    # 没有检测到数据时为None
        for handLms in results.multi_hand_landmarks:  # single hand landmark
            for id, landmark in enumerate(handLms.landmark):  # finger id andlandmark
                # print(id,landmark)
                height, width, c = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                # print(id, cx, cy)
                if id == 0:   # find a point location
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # connections

    # fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS:" + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)