import cv2
from ultralytics import YOLO
import random
import time

rtsp_url = "rtsp://128.14.65.208:8080/h264_ulaw.sdp"


cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

model = YOLO('yolov8n.pt')

def detect_objects(frame):
    results = model(frame)
    return results

def get_random_color():
    while True:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if color != (0, 0, 255):
            return color

colors = {}
selected_box = None
timer_start = None

def on_mouse(event, x, y, flags, param):
    global selected_box, timer_start
    if event == cv2.EVENT_LBUTTONDOWN:
        for bbox in param:
            x1, y1, x2, y2, obj_id = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_box = obj_id
                timer_start = time.time()
                colors[obj_id] = (0, 0, 255)
                break

cv2.namedWindow('RTSP Stream')
cv2.setMouseCallback('RTSP Stream', on_mouse)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detect_objects(frame)
    bboxes = []

    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0: 
                x1, y1, x2, y2 = map(int, box.xyxy)
                obj_id = box.id
                bboxes.append((x1, y1, x2, y2, obj_id))
                if obj_id not in colors:
                    colors[obj_id] = get_random_color()
                color = colors[obj_id]
                if obj_id == selected_box:
                    color = (0, 0, 255)
                    elapsed_time = int(time.time() - timer_start)
                    cv2.putText(frame, f'Timer: {elapsed_time}s', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow('RTSP Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
