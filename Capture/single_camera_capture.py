import cv2
import numpy as np
from datetime import datetime
import os

# Chessboard parameters
number_of_squares_x = 36
number_of_squares_y = 14
number_of_internal_corners_x = number_of_squares_x - 1
number_of_internal_corners_y = number_of_squares_y - 1
SQUARE_SIZE = 5.4/6  # in meters
CAMERA_NAME = 'cam0'

# Prepare object points (3D points of the chessboard)
objp = np.zeros((number_of_internal_corners_x * number_of_internal_corners_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:number_of_internal_corners_x, 0:number_of_internal_corners_y].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Initialize camera
camera = cv2.VideoCapture(1)
if not camera.isOpened():
    print("Error: Could not open camera")
    exit()

print(f"Frame width: {camera.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"Frame height: {camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

# Get frame dimensions
ret, frame = camera.read()
if not ret:
    print("Error: Could not read frame")
    camera.release()
    exit()

# Setup video writer
path = '../photos/single_camera'
os.makedirs(path, exist_ok=True)
# Parameters for image selection
MAX_IMAGES = 300  # Target number of images for calibration
number_of_images = 0

while True:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{path}/{CAMERA_NAME}_{timestamp}.jpg'
    
    ret, frame = camera.read()
    if not ret:
        print("Failed to read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, 
        (number_of_internal_corners_x, number_of_internal_corners_y), 
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        cv2.imwrite(filename, frame)

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or number_of_images >= MAX_IMAGES:
        break
    
    number_of_images += 1

# Release resources
camera.release()
cv2.destroyAllWindows()