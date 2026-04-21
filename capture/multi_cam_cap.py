"""Record a vertically stacked 3-camera video for get_points.py."""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

CAMERA_INDICES = [2, 1, 0]  # top to bottom: cam2, cam1, cam0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 960
FPS = 24
OUTPUT_DIR = Path("../video/charuco_air")

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()


def draw_tag_count(frame):
    _, ids, _ = cv2.aruco.detectMarkers(frame, ARUCO_DICT, parameters=ARUCO_PARAMS)
    count = 0 if ids is None else len(ids)
    cv2.putText(frame, str(count), (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, cv2.LINE_AA)


def main():
    caps = [cv2.VideoCapture(i) for i in CAMERA_INDICES]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: Could not open camera {CAMERA_INDICES[i]}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Grab one frame from each camera to discover the actual resolution.
    first_frames = []
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Error: Could not read from camera {CAMERA_INDICES[i]}")
            return
        first_frames.append(frame)

    shapes = [f.shape[:2] for f in first_frames]
    if len(set(shapes)) != 1:
        print(f"Error: camera frame shapes differ: {shapes}")
        return
    h, w = shapes[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = OUTPUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(filename), fourcc, FPS, (w, h * 3))
    if not out.isOpened():
        print(f"Error: VideoWriter failed to open for {filename}")
        return

    print(f"Recording {w}x{h*3} @ {FPS} to {filename}. Press 'q' to stop.")
    frames = first_frames
    while True:
        out.write(np.vstack(frames))

        preview_frames = [f.copy() for f in frames]
        for f in preview_frames:
            draw_tag_count(f)
        preview = np.vstack(preview_frames)
        preview = cv2.resize(preview, (w // 2, h * 3 // 2))
        cv2.imshow('capture', preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret or frame is None:
                frames = None
                break
            frames.append(frame)
        if frames is None:
            break

    out.release()
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {filename}")


if __name__ == "__main__":
    main()
