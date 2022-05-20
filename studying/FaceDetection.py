import cv2
import mediapipe as mp
import time


mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(imgRGB)
    # print(results)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)  # 坐标数据
            # draw self
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h),
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, "Score: " + f'{int(detection.score[0]*100)}%', (200, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img,"FPS:" + str(int(fps)), (30, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)