"""ChArUco board point extraction — single-camera or multi-camera synchronized."""

import cv2
import numpy as np
from pathlib import Path
import tqdm


def find_images(images_folder):
    """Find all image files in a folder, sorted by name."""
    img_folder = Path(images_folder)
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        img_files.extend(img_folder.glob(ext))
        img_files.extend(img_folder.glob(ext.upper()))
    return sorted(img_files)


def _detect_charuco(gray, board, detector, min_corners):
    """Returns (obj_points, img_points) or (None, None) on failure."""
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    if marker_ids is None or len(marker_ids) == 0:
        return None, None
    num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )
    if num <= min_corners:
        return None, None
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    return obj_points[:, 0, :], img_points[:, 0, :]


def get_points_charuco(
    images_folder,
    output_file,
    camera_names=None,
    charuco_squares_x=11,
    charuco_squares_y=8,
    square_size=0.023,
    marker_size=0.017,
    aruco_dict_name="DICT_5X5_100",
    min_corners=15,
):
    """
    Detect ChArUco corners and save to .npz.

    camera_names=None  → single-camera mode (intrinsic calibration).
    camera_names=list  → multi-camera mode; each image is split vertically
                         into len(camera_names) sub-frames (extrinsic calibration).
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    board = cv2.aruco.CharucoBoard(
        (charuco_squares_x, charuco_squares_y), square_size, marker_size, aruco_dict)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    img_files = find_images(images_folder)
    if not img_files:
        raise FileNotFoundError(f"No images found in {images_folder}")

    multicam = camera_names is not None
    n_cams = len(camera_names) if multicam else 1

    if multicam:
        per_cam = {c: {"obj": [], "img": [], "idx": []} for c in camera_names}
    else:
        obj_list, img_list = [], []

    img_size = None
    print(f"Processing {len(img_files)} images ...")

    for fi, img_file in enumerate(tqdm.tqdm(img_files)):
        full = cv2.imread(str(img_file))
        if full is None:
            continue

        if n_cams == 1:
            sub_frames = [full]
        else:
            h = full.shape[0] // n_cams
            sub_frames = [full[i * h:(i + 1) * h, :] for i in range(n_cams)]

        for ci in range(n_cams):
            sf = sub_frames[ci]
            if img_size is None:
                img_size = (sf.shape[1], sf.shape[0])
            gray = cv2.cvtColor(sf, cv2.COLOR_BGR2GRAY)
            obj, img = _detect_charuco(gray, board, detector, min_corners)
            if obj is None:
                continue
            if multicam:
                cam = camera_names[ci]
                per_cam[cam]["obj"].append(obj)
                per_cam[cam]["img"].append(img)
                per_cam[cam]["idx"].append(fi)
            else:
                obj_list.append(obj)
                img_list.append(img)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    if multicam:
        save = dict(
            camera_names=np.array(camera_names),
            img_size=np.array(img_size),
            board_size=(charuco_squares_x, charuco_squares_y),
            square_size=square_size, marker_size=marker_size,
            total_frames=len(img_files),
        )
        for c in camera_names:
            n = len(per_cam[c]["obj"])
            print(f"  {c}: {n}/{len(img_files)} frames")
            save[f"{c}_obj_points"] = np.array(per_cam[c]["obj"], dtype=object)
            save[f"{c}_img_points"] = np.array(per_cam[c]["img"], dtype=object)
            save[f"{c}_frame_indices"] = np.array(per_cam[c]["idx"], dtype=int)
        np.savez(output_file, **save)
        print(f"Saved {output_file}")
        return save

    print(f"Detected in {len(obj_list)}/{len(img_files)} images")
    if not obj_list:
        raise RuntimeError("No ChArUco boards detected!")
    result = dict(
        obj_points=np.array(obj_list, dtype=object),
        img_points=np.array(img_list, dtype=object),
        img_size=np.array(img_size),
        board_size=(charuco_squares_x, charuco_squares_y),
        square_size=square_size, marker_size=marker_size,
    )
    np.savez(output_file, **result)
    print(f"Saved {output_file}")
    return result
