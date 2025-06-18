import cv2
import numpy as np

# Open the cameras
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

# Get the frame size
height, width = frame_1.shape[:2]

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
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width * 3, height * 2))

while True:
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