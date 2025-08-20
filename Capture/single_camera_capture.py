import cv2
import numpy as np
from datetime import datetime
import os

CAMERA_NAME = 'cam0'

# Initialize camera
camera = cv2.VideoCapture(1)
if not camera.isOpened():
    print("Error: Could not open camera")
    exit()
    
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get frame dimensions
ret, frame = camera.read()
if not ret:
    print("Error: Could not read frame")
    camera.release()
    exit()

# Setup video writer
path = '../video'
os.makedirs(path, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'{path}/{CAMERA_NAME}_{timestamp}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(filename, fourcc, 24, (width * 3, height * 2))

while True:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{path}/{CAMERA_NAME}_{timestamp}.jpg'
    
    ret, frame = camera.read()
    if not ret:
        print("Failed to read frame")
        break

    out.write(frame)
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Release resources
camera.release()
cv2.destroyAllWindows()