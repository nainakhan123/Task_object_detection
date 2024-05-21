import cv2
from ultralytics import YOLO
import random
import time

video_path = "/home/sumbal12/TASK/classroom.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video")
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
selected_boxes = {}
timer_starts = {}

def on_mouse(event, x, y, flags, param):
    global selected_boxes, timer_starts
    if event == cv2.EVENT_LBUTTONDOWN:
        for bbox in param:
            x1, y1, x2, y2, obj_id = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                if obj_id in selected_boxes:
                    colors[obj_id] = get_random_color()
                    timer_starts.pop(obj_id, None)
                selected_boxes[obj_id] = True
                timer_starts[obj_id] = time.time()
                colors[obj_id] = (0, 0, 255)

cv2.namedWindow('Video Stream')
cv2.setMouseCallback('Video Stream', on_mouse, param=[])

tracker = cv2.TrackerMIL_create()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detect_objects(frame)
    bboxes = []

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                obj_id = box.id[0].item() if box.id is not None else None
                bboxes.append((x1, y1, x2, y2, obj_id))
                color = colors.get(obj_id, get_random_color())
                if obj_id in selected_boxes:
                    color = (0, 0, 255) 
                    if obj_id in timer_starts:
                        elapsed_time = int(time.time() - timer_starts[obj_id])
                        cv2.putText(frame, f'Timer: {elapsed_time}s', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow('Video Stream', frame)
    cv2.setMouseCallback('Video Stream', on_mouse, param=bboxes)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
