from ultralytics import YOLO
import cvzone
import cv2
import math
import time

# Running real time from webcam
cap = cv2.VideoCapture(0)
model = YOLO('fire1.pt')

# Reading the classes
classnames = ['fire']

# FPS variables
prev_time = 0

# Fire detection counter
fire_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.resize(frame, (640, 480))

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    result = model(frame, stream=True)

    fire_detected = False

    # Getting bbox, confidence and class names
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = math.ceil(box.conf[0] * 100)
            Class = int(box.cls[0])

            if confidence > 50:
                fire_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Fire bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

                cvzone.putTextRect(
                    frame,
                    f'{classnames[Class]} {confidence}%',
                    [x1 + 8, y1 + 40],
                    scale=1.5,
                    thickness=2
                )

    # 🔥 Fire Alert Feature
    if fire_detected:
        fire_count += 1

        cv2.putText(frame,
                    "FIRE DETECTED!",
                    (170, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3)

        # 📸 Screenshot capture
        filename = f"fire_capture_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)

    # 🔢 Display detection countss
    cv2.putText(frame,
                f'Fire Count: {fire_count}',
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2)

    # ⚡ FPS display
    cv2.putText(frame,
                f'FPS: {int(fps)}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    cv2.imshow('Fire Detection System', frame)

    # Exit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()