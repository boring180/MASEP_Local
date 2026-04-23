"""Multi-camera ChArUco localization via PnP with different intrinsic models.

Supports the three underwater intrinsic methods produced by
`intrinsic/intrinsic_calibration.py`:
  - "air_direct":   air calibration applied as-is to water pixels.
  - "water_opencv": OpenCV recalibration on water points.
  - "refractive":   air K/D plus a learned refractive index n
                    (Singh & Alexis 2024 flat-port model).
"""

import cv2
import numpy as np
import pickle
from pathlib import Path


CAMERA_NAMES = ["cam2", "cam1", "cam0"]  # top to bottom
METHODS = ("air_direct", "water_opencv", "refractive")


def load_extrinsics(calib_dir):
    """Load extrinsic transforms (4x4) for each camera from pickle files."""
    extrinsics = {}
    for cam in CAMERA_NAMES:
        p = Path(calib_dir) / f"extrinsic_{cam}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                extrinsics[cam] = pickle.load(f)
    return extrinsics


def load_intrinsics(calib_dir, method):
    """Load per-camera intrinsics for one of the methods in `METHODS`.

    Returns dict of {cam: {"K", "dist", "n"?}}. `n` is only present for the
    refractive method.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    intrinsics = {}
    for cam in CAMERA_NAMES:
        p = Path(calib_dir) / f"{cam}_{method}.npz"
        if not p.exists():
            continue
        d = np.load(str(p))
        entry = {"K": np.asarray(d["K"], np.float64),
                 "dist": np.asarray(d["dist"], np.float64).reshape(-1)}
        if method == "refractive":
            entry["n"] = float(d["n"])
        intrinsics[cam] = entry
    return intrinsics


def make_charuco_detector(squares_x=11, squares_y=8, square_size=0.023,
                          marker_size=0.017, aruco_dict_name="DICT_5X5_100"):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_size, marker_size, aruco_dict)
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def detect_charuco(gray, board, detector, min_corners=6):
    """Detect ChArUco corners. Returns (obj_points, img_points) or (None, None)."""
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < min_corners:
        return None, None
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    return obj_points[:, 0, :], img_points[:, 0, :]


def _rt_to_pose(rvec, tvec):
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(rvec)
    T[:3, 3] = tvec.flatten()
    return T


def solve_pnp_opencv(obj_pts, img_pts, K, dist):
    """SolvePnP using OpenCV intrinsics (K + dist)."""
    ret, rvec, tvec = cv2.solvePnP(
        obj_pts.astype(np.float32), img_pts.astype(np.float32), K, dist)
    return _rt_to_pose(rvec, tvec) if ret else None


def solve_pnp_refractive(obj_pts, img_pts, K, dist, n):
    """PnP under the Singh & Alexis 2024 flat-port refractive model.

    Pipeline: (1) lens-undistort observed water pixels with air K/D to get
    normalized refracted rays p̄_{C,r}; (2) invert refraction to recover the
    air-equivalent rays p̄_C = p̄_{C,r} / sqrt(n² r_r² + n² − r_r²); (3) map
    back to pixels with air K and solve PnP against zero distortion.
    """
    normalized = cv2.undistortPoints(
        img_pts.astype(np.float32).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    rr2 = (normalized ** 2).sum(axis=1)
    m = np.sqrt(n * n * rr2 + n * n - rr2)
    undist_norm = normalized / m[:, None]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    undist_px = np.stack([undist_norm[:, 0] * fx + cx,
                          undist_norm[:, 1] * fy + cy], axis=1)
    ret, rvec, tvec = cv2.solvePnP(
        obj_pts.astype(np.float32), undist_px.astype(np.float32),
        K, np.zeros(5))
    return _rt_to_pose(rvec, tvec) if ret else None


def detect_frame(sub_frames, board, detector, min_corners=6):
    """Detect ChArUco corners in each camera sub-frame.

    Returns:
        dict of {cam: (obj_points, img_points)} for cameras with detections
    """
    detections = {}
    for ci, cam in enumerate(CAMERA_NAMES):
        gray = cv2.cvtColor(sub_frames[ci], cv2.COLOR_BGR2GRAY)
        obj_pts, img_pts = detect_charuco(gray, board, detector, min_corners)
        if obj_pts is not None:
            detections[cam] = (obj_pts, img_pts)
    return detections


def localize_from_detections(detections, intrinsics_dict, extrinsics,
                             method="air_direct"):
    """Run PnP on pre-detected points under the chosen intrinsic method.

    `intrinsics_dict` is the dict returned by `load_intrinsics(calib_dir, method)`.
    For `air_direct` / `water_opencv` it yields {K, dist}; for `refractive` it
    additionally carries the learned `n`.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    results = {}
    for cam, (obj_pts, img_pts) in detections.items():
        if cam not in intrinsics_dict or cam not in extrinsics:
            continue
        entry = intrinsics_dict[cam]
        if method == "refractive":
            T_cam_board = solve_pnp_refractive(
                obj_pts, img_pts, entry["K"], entry["dist"], entry["n"])
        else:
            T_cam_board = solve_pnp_opencv(
                obj_pts, img_pts, entry["K"], entry["dist"])
        if T_cam_board is None:
            continue
        results[cam] = extrinsics[cam] @ T_cam_board
    return results


def localize_frame(sub_frames, board, detector, intrinsics_dict, extrinsics,
                    method="air_direct", min_corners=6):
    """Localize the charuco board in each camera; return `{cam: world pose}`.

    `method` must be one of `METHODS`; see `localize_from_detections`.
    """
    detections = detect_frame(sub_frames, board, detector, min_corners)
    return localize_from_detections(detections, intrinsics_dict, extrinsics, method)


def detect_video(video_path, board, detector, min_corners=6):
    """Detect ChArUco corners for all frames in a video.

    Returns:
        list of dicts, each {cam: (obj_points, img_points)}
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_detections = []

    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        h = frame.shape[0] // 3
        subs = [frame[i * h:(i + 1) * h, :] for i in range(3)]
        all_detections.append(detect_frame(subs, board, detector, min_corners))

    cap.release()
    return all_detections


def process_video(video_path, board, detector, intrinsics_dict, extrinsics,
                  method="air_direct", min_corners=6):
    """Process a video and return per-frame localization results.

    Returns:
        list of dicts, each {cam: 4x4 world pose}
    """
    all_detections = detect_video(video_path, board, detector, min_corners)
    return [localize_from_detections(det, intrinsics_dict, extrinsics, method)
            for det in all_detections]
