"""Multi-camera ChArUco localization via PnP with different intrinsic models."""

import cv2
import numpy as np
import pickle
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "intrinsic"))
from intrinsic_calibration import pytorch_distortion_fit


CAMERA_NAMES = ["cam2", "cam1", "cam0"]  # top to bottom


def load_extrinsics(calib_dir):
    """Load extrinsic transforms (4x4) for each camera from pickle files."""
    extrinsics = {}
    for cam in CAMERA_NAMES:
        p = Path(calib_dir) / f"extrinsic_{cam}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                extrinsics[cam] = pickle.load(f)
    return extrinsics


def load_opencv_intrinsics(calib_dir):
    """Load OpenCV K and dist for each camera."""
    intrinsics = {}
    for cam in CAMERA_NAMES:
        p = Path(calib_dir) / f"{cam}_opencv.npz"
        if p.exists():
            d = np.load(str(p))
            intrinsics[cam] = {"K": d["K"], "dist": d["dist"]}
    return intrinsics


def load_pytorch_intrinsics(calib_dir, model_suffix):
    """Load PyTorch distortion model (.pt) for each camera.

    model_suffix: 'poly' or 'mlp'
    Returns dict of {cam: pytorch_distortion_fit} with K accessible via .mtx
    """
    intrinsics = {}
    for cam in CAMERA_NAMES:
        p = Path(calib_dir) / f"{cam}_{model_suffix}.pt"
        if p.exists():
            intrinsics[cam] = pytorch_distortion_fit.load(str(p))
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


def solve_pnp_opencv(obj_pts, img_pts, K, dist):
    """SolvePnP using OpenCV intrinsics (K + dist)."""
    ret, rvec, tvec = cv2.solvePnP(
        obj_pts.astype(np.float32), img_pts.astype(np.float32), K, dist)
    if not ret:
        return None
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(rvec)
    T[:3, 3] = tvec.flatten()
    return T


def solve_pnp_pytorch(obj_pts, img_pts, pt_model):
    """SolvePnP using PyTorch distortion model: undistort points first, then PnP with dist=0."""
    undist = pt_model.undistort_points([img_pts])[0]
    ret, rvec, tvec = cv2.solvePnP(
        obj_pts.astype(np.float32), undist.astype(np.float32),
        pt_model.mtx, np.zeros(5))
    if not ret:
        return None
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(rvec)
    T[:3, 3] = tvec.flatten()
    return T


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
                             method="opencv"):
    """Run PnP on pre-detected points for a given intrinsic method.

    Args:
        detections: dict of {cam: (obj_points, img_points)} from detect_frame
        intrinsics_dict: for 'opencv': {cam: {K, dist}}, for 'poly'/'mlp': {cam: pytorch_distortion_fit}
        method: 'opencv', 'poly', or 'mlp'

    Returns:
        dict of {cam: 4x4 world pose} for cameras with valid PnP solutions
    """
    results = {}
    for cam, (obj_pts, img_pts) in detections.items():
        if cam not in intrinsics_dict or cam not in extrinsics:
            continue

        if method == "opencv":
            T_cam_board = solve_pnp_opencv(
                obj_pts, img_pts, intrinsics_dict[cam]["K"], intrinsics_dict[cam]["dist"])
        else:
            T_cam_board = solve_pnp_pytorch(obj_pts, img_pts, intrinsics_dict[cam])

        if T_cam_board is None:
            continue

        T_world_board = np.linalg.inv(extrinsics[cam]) @ T_cam_board
        results[cam] = T_world_board

    return results


def localize_frame(sub_frames, board, detector, intrinsics_dict, extrinsics,
                    method="opencv", min_corners=6):
    """Localize charuco board in each camera, return world poses.

    Args:
        sub_frames: list of 3 camera images (top to bottom)
        intrinsics_dict: for 'opencv': {cam: {K, dist}}, for 'poly'/'mlp': {cam: pytorch_distortion_fit}
        method: 'opencv', 'poly', or 'mlp'

    Returns:
        dict of {cam: 4x4 world pose} for cameras that detected the board
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
                  method="opencv", min_corners=6):
    """Process a video and return per-frame localization results.

    Returns:
        list of dicts, each {cam: 4x4 world pose}
    """
    all_detections = detect_video(video_path, board, detector, min_corners)
    return [localize_from_detections(det, intrinsics_dict, extrinsics, method)
            for det in all_detections]
