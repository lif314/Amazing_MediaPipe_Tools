import cv2
import mediapipe as mp
import time

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# 自定义渲染样式
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:  # one face
            mpDraw.draw_landmarks(img, faceLms,
                                  mpFaceMesh.FACE_CONNECTIONS,
                                  connection_drawing_spec=drawSpec)
            for id, lm in enumerate(faceLms.landmark):  # each point
                h, w, c = img.shape
                cx, cy = int(w * lm.x), int(h * lm.y)
                if id == 0:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "FPS:" + str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
