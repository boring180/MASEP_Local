"""ChArUco board point extraction for intrinsic and extrinsic calibration."""

import cv2
import numpy as np
import os
from pathlib import Path
import tqdm

CAMERA_NAMES = ["cam2", "cam1", "cam0"]  # top to bottom


def _make_detector(squares_x, squares_y, square_size, marker_size, aruco_dict_name):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_size, marker_size, aruco_dict)
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def _detect_charuco(gray, board, detector, min_corners):
    """Returns (obj_points, img_points) or (None, None) on failure."""
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) <= min_corners:
        return None, None
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    return obj_points[:, 0, :], img_points[:, 0, :]


def _draw_corners(frame, img_points):
    corners = img_points.reshape(-1, 1, 2).astype(np.float32)
    ids = np.arange(len(img_points)).reshape(-1, 1)
    cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids)


def process_videos(
    video_folder,
    points_folder,
    photos_folder,
    capture_per_frame=1,
    charuco_squares_x=11,
    charuco_squares_y=8,
    square_size=0.023,
    marker_size=0.017,
    aruco_dict_name="DICT_5X5_100",
    min_corners=15,
):
    """
    Extract ChArUco points from videos with 3 vertically stacked cameras.

    Produces 4 point files per video:
      - {video}_cam0.npz, {video}_cam1.npz, {video}_cam2.npz  (intrinsic)
      - {video}_extrinsic.npz  (synced frames where >=2 cameras detected)

    Photos saved under:
      - photos_folder/single/   (frames with any detection, per-camera)
      - photos_folder/multi/    (frames where >=2 cameras detected, combined)
    """
    video_dir = Path(video_folder)
    video_files = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'))
    if not video_files:
        raise FileNotFoundError(f"No video files in {video_folder}")

    board, detector = _make_detector(
        charuco_squares_x, charuco_squares_y, square_size, marker_size, aruco_dict_name)

    points_dir = Path(points_folder)
    single_photo_dir = Path(photos_folder) / "single"
    multi_photo_dir = Path(photos_folder) / "multi"
    for d in [points_dir, single_photo_dir, multi_photo_dir]:
        if d.exists():
            for f in d.iterdir():
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    board_params = dict(
        board_size=(charuco_squares_x, charuco_squares_y),
        square_size=square_size, marker_size=marker_size)

    for video_file in video_files:
        print(f"Processing {video_file.name} ...")
        cap = cv2.VideoCapture(str(video_file))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stem = video_file.stem

        # Per-camera intrinsic points
        intrinsic = {c: {"obj": [], "img": []} for c in CAMERA_NAMES}
        # Extrinsic synced points
        extrinsic = {c: {"obj": [], "img": []} for c in CAMERA_NAMES}
        img_size = None
        frame_idx = 0

        for _ in tqdm.tqdm(range(total_frames), desc=stem):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % capture_per_frame != 0:
                frame_idx += 1
                continue

            h = frame.shape[0] // 3
            subs = [frame[i * h:(i + 1) * h, :].copy() for i in range(3)]
            if img_size is None:
                img_size = (subs[0].shape[1], subs[0].shape[0])

            # Detect in each camera
            detections = {}
            for ci, cam in enumerate(CAMERA_NAMES):
                gray = cv2.cvtColor(subs[ci], cv2.COLOR_BGR2GRAY)
                obj, img = _detect_charuco(gray, board, detector, min_corners)
                if obj is not None:
                    detections[cam] = (obj, img, ci)

            # Save intrinsic points and single-camera photos
            for cam, (obj, img, ci) in detections.items():
                intrinsic[cam]["obj"].append(obj)
                intrinsic[cam]["img"].append(img)
                _draw_corners(subs[ci], img)
                photo_path = single_photo_dir / f"{stem}_{frame_idx}_{cam}.jpg"
                cv2.imwrite(str(photo_path), subs[ci])

            # Save extrinsic points and multi-camera photo when >=2 cameras detect
            if len(detections) >= 2:
                for cam in CAMERA_NAMES:
                    if cam in detections:
                        obj, img, _ = detections[cam]
                        extrinsic[cam]["obj"].append(obj)
                        extrinsic[cam]["img"].append(img)
                    else:
                        extrinsic[cam]["obj"].append(None)
                        extrinsic[cam]["img"].append(None)
                combined = np.vstack(subs)
                cv2.imwrite(str(multi_photo_dir / f"{stem}_{frame_idx}.jpg"), combined)

            frame_idx += 1

        cap.release()

        # Save intrinsic point files (one per camera)
        for cam in CAMERA_NAMES:
            n = len(intrinsic[cam]["obj"])
            print(f"  {cam} intrinsic: {n} frames")
            if n > 0:
                np.savez(str(points_dir / f"{stem}_{cam}.npz"),
                         obj_points=np.array(intrinsic[cam]["obj"], dtype=object),
                         img_points=np.array(intrinsic[cam]["img"], dtype=object),
                         img_size=np.array(img_size), **board_params)

        # Save extrinsic point file
        n_ext = len(extrinsic[CAMERA_NAMES[0]]["obj"])
        print(f"  extrinsic: {n_ext} synced frames")
        if n_ext > 0:
            ext_save = dict(camera_names=np.array(CAMERA_NAMES),
                            img_size=np.array(img_size), **board_params)
            for cam in CAMERA_NAMES:
                ext_save[f"{cam}_obj_points"] = np.array(extrinsic[cam]["obj"], dtype=object)
                ext_save[f"{cam}_img_points"] = np.array(extrinsic[cam]["img"], dtype=object)
            np.savez(str(points_dir / f"{stem}_extrinsic.npz"), **ext_save)


if __name__ == "__main__":
    process_videos(
        video_folder="../video/charuco_air",
        points_folder="../points/air",
        photos_folder="../photos/air",
    )
