import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

cap = cv2.VideoCapture(0)
prev_time = time.perf_counter()

base_options = python.BaseOptions(model_asset_path = "face_landmarker.task")
options = vision.FaceLandmarkerOptions(base_options = base_options,running_mode = vision.RunningMode.VIDEO)
landmarker = vision.FaceLandmarker.create_from_options(options)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read from camera")
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        h, w, _ = frame.shape
        for point in landmarks:
            x = int(point.x * w)
            y = int(point.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    now = time.perf_counter()
    fps = 1 / (now - prev_time)
    prev_time = now

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()   