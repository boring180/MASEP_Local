import cv2
import numpy as np
from datetime import datetime
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath('.')))
from utils.frame_slicing import slicing_frame
from utils.frame_concatent import concatent_frame
from queue import Queue
import threading

# Chessboard pattern settings for calibration
COL_NUM = 11
ROW_NUM = 8
CHESSBOARD_SIZE = (COL_NUM - 1, ROW_NUM - 1)  # Number of inner corners (columns, rows)
SQUARE_SIZE = 0.023  # Size of a square in meters
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

def main():
    # Initialize cameras (only first three)
    cameras = [cv2.VideoCapture(0), cv2.VideoCapture(1), cv2.VideoCapture(3)]
    
    # Verify cameras opened successfully
    for i in range(len(cameras)):
        if not cameras[i].isOpened():
            print(f"Error: Could not open camera {i}")
            return
    
    frames = []
    for i in range(len(cameras)):
        ret, frame = cameras[i].read()
        frames.append(frame)
    print(f'Frame number: {len(frames)}')
    frame = concatent_frame(frames)
    height, width = frame.shape[:2]

    # Create output directory
    if not os.path.exists('output'):
        os.makedirs('output')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'output/{timestamp}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 24, (width, height))

    # Frame rate monitoring
    frame_count = 0
    start_time = time.time()

    print("Press 'q' to stop recording")

    while True:
        frames = []

        # Capture frames from all cameras
        for i in range(len(cameras)):
            ret, frame = cameras[i].read()
            if not ret:
                print(f"Warning: Could not read frame from camera {i}")
            frames.append(frame)
        
        show_frames = []
        for i in range(len(frames)):
            shown_frame = frames[i].copy()
            gray = cv2.cvtColor(shown_frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            if ret:
                cv2.drawChessboardCorners(shown_frame, CHESSBOARD_SIZE, corners, ret)
            show_frames.append(shown_frame)
            
        # Write original frames (without chessboard) to video
        out.write(concatent_frame(frames))
        cv2.imshow('Cameras with Chessboard', concatent_frame(show_frames))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            current_fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            print(f"Current FPS: {current_fps:.2f}")

    # Cleanup
    out.release()
    for _, cap in cameras:
        cap.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {filename}")
    
if __name__ == "__main__":
    main()