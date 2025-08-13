import cv2
import numpy as np
from datetime import datetime
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath('.')))

# Chessboard pattern settings for calibration
COL_NUM = 11
ROW_NUM = 8
CHESSBOARD_SIZE = (COL_NUM - 1, ROW_NUM - 1)  # Number of inner corners (columns, rows)
SQUARE_SIZE = 0.023  # Size of a square in meters
FLAGS = cv2.CALIB_CB_FAST_CHECK 

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

def frame_concatenate(frames, reference_shape):
    for i in range(len(frames)):
        frames[i] = cv2.resize(frames[i], (reference_shape[1], reference_shape[0]))
    return np.concatenate(frames, axis=1)

def main():
    # Initialize cameras (only first three)
    cameras = [cv2.VideoCapture(0)]
    reference_shape = cameras[0].read()[1].shape[:2]
    
    # Verify cameras opened successfully
    for i in range(len(cameras)):
        if not cameras[i].isOpened():
            print(f"Error: Could not open camera {i}")
            return
    
    frames = []
    for i in range(len(cameras)):
        ret, frame = cameras[i].read()
        frames.append(frame)

    frame = frame_concatenate(frames, reference_shape)
    height, width = frame.shape[:2]

    # Create output directory
    if not os.path.exists('output'):
        os.makedirs('output')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'output/{timestamp}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 24, (width, height))

    prev_time = time.time()

    while True:
        frames = []

        # Capture frames from all cameras
        for i in range(len(cameras)):
            ret, frame = cameras[i].read()
            frames.append(frame)
        
        show_frames = []
        for i in range(len(frames)):
            shown_frame = frames[i].copy()
            gray = cv2.cvtColor(shown_frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None, FLAGS)
            if ret:
                text = "Detected"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 15
                thickness = 30
                color = (0, 0, 255)
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                center_x = shown_frame.shape[1] // 2
                center_y = shown_frame.shape[0] // 2
                org = (center_x - text_width // 2, center_y + text_height // 2)
                cv2.putText(shown_frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
            show_frames.append(shown_frame)
            
        # Write original frames (without chessboard) to video
        # out.write(concatent_frame(frames))
        
        time_elapsed = time.time() - prev_time
        FPS = 1 / time_elapsed
        print(f"FPS: {FPS}")
        prev_time = time.time()
        
        cv2.imshow('Frames', frame_concatenate(show_frames, reference_shape))
        
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