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
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners (columns, rows)
SQUARE_SIZE = 0.025  # Size of a square in meters
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

def put_chessboard_corners(frame_queue, display_frames_queue):
    # Create display frame with chessboard corners
    while True:
        if not frame_queue.empty():
            frames = frame_queue.get()
            if frames is None:
                break
            for i in range(len(frames)):
                frame = frames[i]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
                if ret:
                    cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret)
                frames[i] = frame
            display_frames_queue.put(concatent_frame(frames))
        else:
            time.sleep(0.01)
            continue

def main():
    # Initialize cameras (only first three)
    cameras = [cv2.VideoCapture(1), cv2.VideoCapture(2), cv2.VideoCapture(3)]
    
    # Verify cameras opened successfully
    for i in range(len(cameras)):
        if not cameras[i].isOpened():
            print(f"Error: Could not open camera {i}")
            return
        
    frame_queue = Queue()
    display_frames_queue = Queue()
    
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
    
    chessboard_thread = threading.Thread(target=put_chessboard_corners, args=(frame_queue, display_frames_queue))
    chessboard_thread.start()

    while True:
        frames = []
        display_frames = []

        # Capture frames from all cameras
        for i in range(len(cameras)):
            ret, frame = cameras[i].read()
            if not ret:
                print(f"Warning: Could not read frame from camera {i}")
            frames.append(frame)
            frame_queue.put(frames)
            
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
    
    frame_queue.put(None)
    
    chessboard_thread.join()
    if chessboard_thread.is_alive():
        chessboard_thread.terminate()
    
if __name__ == "__main__":
    main()