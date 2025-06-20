import cv2
import numpy as np
from datetime import datetime
import os
import time

# Target frame rate
TARGET_FPS = 20.0
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Open the cameras
camera_1 = cv2.VideoCapture(1) # Camera 2
print("Camera1 open")
camera_2 = cv2.VideoCapture(2) # Camera 3
print("Camera2 open")
camera_3 = cv2.VideoCapture(3) # Camera middle
print("Camera3 open")
camera_4 = cv2.VideoCapture(4) # Camera 0
print("Camera4 open")
camera_5 = cv2.VideoCapture(5) # Camera 1
print("Camera5 open")

# Configure cameras for consistent frame rate
cameras = [camera_1, camera_2, camera_3, camera_4, camera_5]
for camera in cameras:
    # Set camera properties for consistent frame rate
    camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set a standard width
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set a standard height
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency

# Read one frame from each camera to determine frame size
ret1, frame_1 = camera_1.read()
ret2, frame_2 = camera_2.read()
ret3, frame_3 = camera_3.read()
ret4, frame_4 = camera_4.read()
ret5, frame_5 = camera_5.read()

# Get the frame size
height, width = frame_1.shape[:2]
print(f"Frame size: {width}x{height}")

# Concatenate the frames
def frame_concatenate(frame_1, frame_2, frame_3, frame_4, frame_5):    
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    row1 = np.concatenate((frame_1, frame_2, frame_3), axis=1)
    row2 = np.concatenate((frame_4, frame_5, black_frame), axis=1)
    frame = np.concatenate((row1, row2), axis=0)
    return frame

# Frame resize
def frame_resize(frame, width, height):
    return cv2.resize(frame, (width, height))

# Define the codec and create VideoWriter object
if not os.path.exists('captures'):
    os.makedirs('captures')
            
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'captures/{timestamp}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(filename, fourcc, TARGET_FPS, (width * 3, height * 2))

# Frame rate monitoring
frame_count = 0
start_time = time.time()

# Create a video writer
while True:
    loop_start_time = time.time()
    
    ret1, frame_1 = camera_1.read()
    ret2, frame_2 = camera_2.read()
    ret3, frame_3 = camera_3.read()
    ret4, frame_4 = camera_4.read()
    ret5, frame_5 = camera_5.read()
    
    # If the frame is not read, use a black frame
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    if not ret1:
        frame_1 = black_frame.copy()
    if not ret2:
        frame_2 = black_frame.copy()
    if not ret3:
        frame_3 = black_frame.copy()
    if not ret4:
        frame_4 = black_frame.copy()
    if not ret5:
        frame_5 = black_frame.copy()
    
    # Resize all frames to the size of frame_1 and concatenate them
    frame_1 = frame_resize(frame_1, width, height)
    frame_2 = frame_resize(frame_2, width, height)
    frame_3 = frame_resize(frame_3, width, height)
    frame_4 = frame_resize(frame_4, width, height)
    frame_5 = frame_resize(frame_5, width, height)
    frame = frame_concatenate(frame_1, frame_2, frame_3, frame_4, frame_5)

    cv2.imshow('Cameras', frame)
    out.write(frame)  # Write the frame to the video file
    
    frame_count += 1
    
    # Calculate and display current FPS
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        current_fps = frame_count / elapsed_time
        print(f"Current FPS: {current_fps:.2f}, Target FPS: {TARGET_FPS}")
    
    # Frame rate control - wait to maintain target FPS
    processing_time = time.time() - loop_start_time
    sleep_time = max(0, FRAME_INTERVAL - processing_time)
    if sleep_time > 0:
        time.sleep(sleep_time)
        
    print("Frame count: ", frame_count)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
out.release()
camera_1.release()
camera_2.release()
camera_3.release()
camera_4.release()
camera_5.release()  
cv2.destroyAllWindows()

# Print final statistics
total_time = time.time() - start_time
final_fps = frame_count / total_time if total_time > 0 else 0
print(f"Recording completed. Total frames: {frame_count}, Total time: {total_time:.2f}s, Average FPS: {final_fps:.2f}")