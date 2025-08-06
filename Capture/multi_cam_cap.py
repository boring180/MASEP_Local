import cv2
import numpy as np
from datetime import datetime
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath('.')))
from utils.frame_slicing import frame_slicing
from utils.frame_concatent import concatent_frame
from queue import Queue
import threading

# Chessboard pattern settings for calibration
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners (columns, rows)
SQUARE_SIZE = 0.025  # Size of a square in meters
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

def display_video(frame_queue, display_frames_queue):
    # Create display frame with chessboard corners
    while True:
        if not frame_queue.empty():
            frames = frame_queue.get()
            display_frames_queue.put(concatent_frame(frames))
        else:
            time.sleep(0.01)
            continue
        
        # display_frame = frame.copy()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        # if ret:
        #     corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        #     cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret)
        # display_frames.append(display_frame)

def main():
    # Initialize cameras (only first three)
    cameras = [cv2.VideoCapture(1), cv2.VideoCapture(2), cv2.VideoCapture(3)]
    
    # Verify cameras opened successfully
    for idx, cap in cameras:
        if not cap.isOpened():
            print(f"Error: Could not open camera {idx}")
            return
        
    frame_queue = Queue()
    display_frames_queue = Queue()
    
    frames = []
    for idx, cap in cameras:
        ret, frame = cap.read()
        frames.append(frame)
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
    
    display_thread = threading.Thread(target=display_video, args=(frame_queue, display_frames_queue))
    display_thread.start()

    while True:
        frames = []
        display_frames = []

        # Capture frames from all cameras
        for idx, cap in cameras:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame from camera {idx}")
            frames.append(frame)
            frame_queue.put(frame)
            
        # Write original frames (without chessboard) to video
        out.write(concatent_frame(frames))
        
        display_frame = display_frames_queue.get()
        cv2.imshow('Cameras with Chessboard', display_frame)

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            current_fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            print(f"Current FPS: {current_fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    out.release()
    for _, cap in cameras:
        cap.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {filename}")
    
if __name__ == "__main__":
    main()