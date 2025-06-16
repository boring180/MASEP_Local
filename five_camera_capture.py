import cv2
import numpy as np

camera_1 = cv2.VideoCapture(0)
camera_2 = cv2.VideoCapture(1)
camera_3 = cv2.VideoCapture(2)
camera_4 = cv2.VideoCapture(3)
camera_5 = cv2.VideoCapture(4)

# Read one frame from each camera to determine frame size
ret1, frame_1 = camera_1.read()
ret2, frame_2 = camera_2.read()
ret3, frame_3 = camera_3.read()
ret4, frame_4 = camera_4.read()
ret5, frame_5 = camera_5.read()

# Concatenate to get the final frame size
black_frame = np.zeros_like(frame_1)
frame1_height, frame1_width = frame_1.shape[:2]
row1 = np.concatenate((frame_1, frame_2, frame_3), axis=1)
row2 = np.concatenate((frame_4, frame_5, black_frame), axis=1)
sample_frame = np.concatenate((row1, row2), axis=0)
height, width, channels = sample_frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

while True:
    ret1, frame_1 = camera_1.read()
    ret2, frame_2 = camera_2.read()
    ret3, frame_3 = camera_3.read()
    ret4, frame_4 = camera_4.read()
    ret5, frame_5 = camera_5.read()
    
    black_frame = np.zeros((frame1_height, frame1_width, 3), dtype=np.uint8)
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
    
    # Resize all frames to the size of frame_1
    frame_2 = cv2.resize(frame_2, (frame1_width, frame1_height))
    frame_3 = cv2.resize(frame_3, (frame1_width, frame1_height))
    frame_4 = cv2.resize(frame_4, (frame1_width, frame1_height))
    frame_5 = cv2.resize(frame_5, (frame1_width, frame1_height))
    black_frame = cv2.resize(black_frame, (frame1_width, frame1_height))
    
    row1 = np.concatenate((frame_1, frame_2, frame_3), axis=1)
    row2 = np.concatenate((frame_4, frame_5, black_frame), axis=1)
    frame = np.concatenate((row1, row2), axis=0)

    cv2.imshow('Camera 1', frame)
    out.write(frame)  # Write the frame to the video file

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