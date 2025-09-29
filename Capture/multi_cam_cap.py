import cv2
import numpy as np
from datetime import datetime
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath('.')))

from settings_loader import settings

# Chessboard pattern settings from settings loader
PATTERN_SIZE = settings.pattern_size
ROW_NUM, COL_NUM = PATTERN_SIZE
CHESSBOARD_SIZE = (COL_NUM - 1, ROW_NUM - 1)  # Number of inner corners (columns, rows)
FLAGS = getattr(cv2, settings.chessboard_flags)

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

### ----------------------------- Charuco pattern settings ----------------------------- ###
SQUARE_SIZE = settings.pattern_square_size  # Size of a square in meters
MARKER_SIZE = settings.marker_size
DICT = cv2.aruco.getPredefinedDictionary(settings.aruco_dict)
board = cv2.aruco.CharucoBoard((settings.pattern_size[0], settings.pattern_size[1]), SQUARE_SIZE, MARKER_SIZE, DICT)
parameters = cv2.aruco.DetectorParameters()
    
def frame_concatent(frames, reference_shape):
    for i in range(len(frames)):
        frames[i] = cv2.resize(frames[i], (reference_shape[1], reference_shape[0]))
    return np.concatenate(frames, axis=1)

def chessboard_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None, FLAGS)
    if ret:
        text = "Detected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 15
        thickness = 30
        color = (0, 0, 255)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        org = (center_x - text_width // 2, center_y + text_height // 2)
        cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
        
def charuco_detection(frame):
    charuco_ids = []
    charuco_corners = []
    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, rejected = cv2.aruco.detectMarkers(frame, DICT, parameters=parameters)
    
    if charuco_ids is not None:
        text = f"{len(charuco_ids)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 15
        thickness = 30
        color = (0, 0, 255)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        org = (center_x - text_width // 2, center_y + text_height // 2)
        cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

def main():
    cameras = [cv2.VideoCapture(1), cv2.VideoCapture(3), cv2.VideoCapture(2)]
    reference_shape = cameras[0].read()[1].shape[:2]
    frames = []
    
    for i in range(len(cameras)):
        cameras[i].set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cameras[i].set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        if not cameras[i].isOpened():
            print(f"Error: Could not open camera {i}")
            return
        ret, frame = cameras[i].read()
        frames.append(frame)

    frame = frame_concatent(frames, reference_shape)
    height, width = frame.shape[:2]

    if not os.path.exists('output'):
        os.makedirs('output')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'output/{timestamp}.mp4'
    if len(settings.cameras) == 1:
        filename = f'output/{settings.cameras[0]}_{timestamp}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 24, (width, height))

    prev_time = time.time()

    while True:
        frames = []
        
        show_frames = []
        for i in range(len(cameras)):
            ret, frame = cameras[i].read()
            frames.append(frame)
            shown_frame = frames[i].copy()
            if settings.pattern_type == 'chessboard':
                chessboard_detection(shown_frame)
            elif settings.pattern_type == 'charuco':
                charuco_detection(shown_frame)
            show_frames.append(shown_frame)
            
        out.write(frame_concatent(frames, reference_shape))
        
        time_elapsed = time.time() - prev_time
        FPS = 1 / time_elapsed
        print(f"FPS: {FPS}")
        prev_time = time.time()
        
        cv2.imshow('Frames', frame_concatent(show_frames, reference_shape))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    for i in range(len(cameras)):
        cameras[i].release()
    cv2.destroyAllWindows()
    
    print(f"Video saved as {filename}")
    
if __name__ == "__main__":
    main()