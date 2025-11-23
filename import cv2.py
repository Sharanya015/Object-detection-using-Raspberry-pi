import os
import sys
import time

try: 
    import cv2
    import numpy as np
except ModuleNotFoundError:
    print("Error: OpenCV (cv2) and NumPy are required but not installed.")
    sys.exit(1)

# Check if YOLO files exist
def check_yolo_files():
    required_files = ["D:\sharanya\major\yolov3.weights", "D:\sharanya\major\yolov3.cfg", "D:\sharanya\major\coco.names"]
    for file in required_files:
        if not os.path.isfile(file):
            print(f"Missing required file: {file}. Please download it.")
            sys.exit(1)

check_yolo_files()

# Load YOLO model
try:
    net = cv2.dnn.readNet("D:\sharanya\major\yolov3.weights", "D:\sharanya\major\yolov3.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    sys.exit(1)

# Load COCO class labels
try:
    with open("D:\sharanya\major\coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Error: COCO class labels file not found.")
    sys.exit(1)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)  # Increase frame rate
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
if not cap.isOpened():
    print("Error: Could not open webcam. Ensure it is properly connected.")
    sys.exit(1)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    height, width, _ = frame.shape
    
    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if isinstance(indexes, np.ndarray):
        indexes = indexes.flatten()
    else:
        indexes = []
    
    for i in indexes:
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the output
    cv2.imshow("Object Detection", frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Calculate and display FPS
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    print(f"FPS: {fps:.2f}")

cap.release()
cv2.destroyAllWindows()