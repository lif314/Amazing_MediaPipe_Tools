import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose(  # 默认参数
               static_image_mode=False,
               upper_body_only=False,
               smooth_landmarks=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5)

mpDraw = mp.solutions.drawing_utils  # draw results


cap = cv2.VideoCapture('./video/yoga.mp4')
# cap = cv2.VideoCapture(0)
pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, connections=mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(w * lm.x), int(h * lm.y)
            # print(id, cx, cy)
            cv2.circle(img, (cx, cy), 2, (255, 0, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)