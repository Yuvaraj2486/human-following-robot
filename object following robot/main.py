import cv2
import numpy as np
from gpiozero import Motor
import time

# Define pins for Motor 1
ENA = 17 # Enable pin for Motor 1
IN1 = 18 # Input 1 pin for Motor 1
IN2 = 27 # Input 2 pin for Motor 1

# Define pins for Motor 2
ENB = 22 # Enable pin for Motor 2
IN3 = 23 # Input 1 pin for Motor 2
IN4 = 24 # Input 2 pin for Motor 2

# Create Motor instances
motor1 = Motor(forward=IN1, backward=IN2, enable=ENA, pwm=True)
motor2 = Motor(forward=IN3, backward=IN4, enable=ENB, pwm=True)

# Function to move the robot towards the detected object
def move_towards_object(x_deviation):
    global speed
    if abs(x_deviation) < tolerance:
        # Object centered, move forward
        motor1.forward(speed)
        motor2.forward(speed)
    else:
        # Object not centered, adjust direction
        if x_deviation > 0:
            # Object on the right, turn right
            motor1.backward(speed)
            motor2.forward(speed)
        else:
            # Object on the left, turn left
            motor1.forward(speed)
            motor2.backward(speed)

# Object detection parameters
threshold = 0.5
tolerance = 0.1
speed = 0.5 # Adjust speed as needed

# Load YOLOv3 model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []

with open('coco.names.txt', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Set target object
target_object = 'person'

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > threshold and classes[class_id] == target_object:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.4)

    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        x_center = x + (w / 2)
        deviation = (x_center - (width / 2)) / (width / 2)
        move_towards_object(deviation)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
