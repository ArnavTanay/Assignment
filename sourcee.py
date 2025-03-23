import cv2
import numpy as np
from imutils.video import VideoStream
import time

# It will load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# For enabling GPU (prevents lag)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Use CPU if no GPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# It will load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# To get layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# For starting the webcam of my PC
cap = VideoStream(src=0).start()
time.sleep(2.0)  # 2 seconds to open (warm up the camera)

frame_id = 0

while True:
    frame = cap.read()
    frame_id += 1

    if frame is None:
        break

    height, width, _ = frame.shape

    # Convert frame to blob (Use smaller size for speed)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Process YOLO output
    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # You need to draw bounding boxes
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Result
    cv2.imshow("YOLO Object Detection", frame)

    # Quit if 'x' is pressed
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.stop()
cv2.destroyAllWindows()