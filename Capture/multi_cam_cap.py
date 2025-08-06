import cv2
import numpy as np
from datetime import datetime
import os
import time

# Chessboard pattern settings for calibration
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners (columns, rows)
SQUARE_SIZE = 0.025  # Size of a square in meters
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def generate_chessboard_points():
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    return objp * SQUARE_SIZE

# Frame concatenation function (3x2 grid with 3 cameras in top row)
def frame_concatenate(frame_1, frame_2, frame_3, width, height):
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    row1 = np.concatenate((frame_1, frame_2, frame_3), axis=1)
    row2 = np.concatenate((black_frame, black_frame, black_frame), axis=1)
    return np.concatenate((row1, row2), axis=0)

# Frame resize function
def frame_resize(frame, width, height):
    return cv2.resize(frame, (width, height))

def main():
    # Initialize cameras (only first three)
    cameras = [
        (1, cv2.VideoCapture(1)),  
        (2, cv2.VideoCapture(2)),  
        (3, cv2.VideoCapture(3))   
    ]
    
    # Verify cameras opened successfully
    for idx, cap in cameras:
        if not cap.isOpened():
            print(f"Error: Could not open camera {idx}")
            return
        # Set consistent frame size (optional, adjust as needed)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Read one frame to determine size
    ret, frame = cameras[0][1].read()
    if not ret:
        print("Error: Could not read frame from camera 1")
        return
    height, width = frame.shape[:2]
    print(f"Frame size: {width}x{height}")

    # Create output directory
    if not os.path.exists('captures'):
        os.makedirs('captures')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'captures/{timestamp}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 24, (width * 3, height * 2))

    # Chessboard points
    objp = generate_chessboard_points()

    # Frame rate monitoring
    frame_count = 0
    start_time = time.time()

    print("Press 'q' to stop recording")

    while True:
        frames = []
        display_frames = []
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Capture frames from all cameras
        for idx, cap in cameras:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame from camera {idx}")
                frame = black_frame.copy()
            frame = frame_resize(frame, width, height)
            frames.append(frame)

            # Create display frame with chessboard corners
            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            if ret:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
                cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret)
            display_frames.append(display_frame)

        # Pad with black frames to maintain 3x2 layout
        while len(frames) < 3:
            frames.append(black_frame.copy())
            display_frames.append(black_frame.copy())

        # Concatenate frames for display and output
        display_frame = frame_concatenate(*display_frames[:3], width, height)
        output_frame = frame_concatenate(*frames[:3], width, height)

        # Show preview with chessboard
        cv2.imshow('Cameras with Chessboard', display_frame)
        # Write original frames (without chessboard) to video
        out.write(output_frame)

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            current_fps = frame_count / elapsed_time
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