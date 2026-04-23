"""ChArUco board point extraction for intrinsic and extrinsic calibration."""

import cv2
import numpy as np
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
    min_corners=1,
    exclusive=False,
):
    """
    Extract ChArUco points from videos with 3 vertically stacked cameras.

    Produces 4 point files per video:
      - {video}_cam0.npz, {video}_cam1.npz, {video}_cam2.npz  (intrinsic)
      - {video}_extrinsic.npz  (synced frames where >=2 cameras detected)

    Photos saved under:
      - photos_folder/single/   (frames with any detection, per-camera)
      - photos_folder/multi/    (frames where >=2 cameras detected, combined)

    If `exclusive` is True, a frame's detections are routed to either the
    intrinsic or the extrinsic set but never both: multi-cam frames (>=2
    detections) go only to extrinsic, single-cam frames go only to intrinsic.
    If False (default), multi-cam frames contribute to both sets.
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

    # Accumulate across all videos
    intrinsic = {c: {"obj": [], "img": []} for c in CAMERA_NAMES}
    extrinsic = {c: {"obj": [], "img": []} for c in CAMERA_NAMES}
    img_size = None
    global_idx = 0

    for video_file in video_files:
        print(f"Processing {video_file.name} ...")
        cap = cv2.VideoCapture(str(video_file))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stem = video_file.stem
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

            detections = {}
            for ci, cam in enumerate(CAMERA_NAMES):
                gray = cv2.cvtColor(subs[ci], cv2.COLOR_BGR2GRAY)
                obj, img = _detect_charuco(gray, board, detector, min_corners)
                if obj is not None:
                    detections[cam] = (obj, img, ci)

            is_multi = len(detections) >= 2
            add_to_intrinsic = not (exclusive and is_multi)

            if add_to_intrinsic:
                for cam, (obj, img, ci) in detections.items():
                    intrinsic[cam]["obj"].append(obj)
                    intrinsic[cam]["img"].append(img)
                    _draw_corners(subs[ci], img)
                    cv2.imwrite(str(single_photo_dir / f"{global_idx}_{cam}.jpg"), subs[ci])

            if is_multi:
                for cam in CAMERA_NAMES:
                    if cam in detections:
                        obj, img, _ = detections[cam]
                        extrinsic[cam]["obj"].append(obj)
                        extrinsic[cam]["img"].append(img)
                    else:
                        extrinsic[cam]["obj"].append(None)
                        extrinsic[cam]["img"].append(None)
                cv2.imwrite(str(multi_photo_dir / f"{global_idx}.jpg"), np.vstack(subs))

            frame_idx += 1
            global_idx += 1

        cap.release()

    # Save one npz per camera (intrinsic)
    for cam in CAMERA_NAMES:
        n = len(intrinsic[cam]["obj"])
        print(f"  {cam} intrinsic: {n} frames")
        if n > 0:
            np.savez(str(points_dir / f"{cam}.npz"),
                     obj_points=np.array(intrinsic[cam]["obj"], dtype=object),
                     img_points=np.array(intrinsic[cam]["img"], dtype=object),
                     img_size=np.array(img_size), **board_params)

    # Save one extrinsic npz
    n_ext = len(extrinsic[CAMERA_NAMES[0]]["obj"])
    print(f"  extrinsic: {n_ext} synced frames")
    if n_ext > 0:
        ext_save = dict(camera_names=np.array(CAMERA_NAMES),
                        img_size=np.array(img_size), **board_params)
        for cam in CAMERA_NAMES:
            ext_save[f"{cam}_obj_points"] = np.array(extrinsic[cam]["obj"], dtype=object)
            ext_save[f"{cam}_img_points"] = np.array(extrinsic[cam]["img"], dtype=object)
        np.savez(str(points_dir / "extrinsic.npz"), **ext_save)


def save_density_heatmaps(points_folder, bin_size=20, blur_sigma=1.5):
    """Render per-camera point-density heatmaps from saved intrinsic npz files.

    Bins all observed image points on a grid of `bin_size`-pixel cells, applies
    a Gaussian blur, and writes a JET-colormapped PNG per camera into
    `points_folder`. Useful to spot regions of the sensor with little/no
    calibration coverage.
    """
    points_dir = Path(points_folder)
    for npz in sorted(points_dir.glob("cam*.npz")):
        data = np.load(str(npz), allow_pickle=True)
        img_w, img_h = (int(x) for x in data["img_size"])
        pts = np.concatenate(
            [np.asarray(p, np.float32).reshape(-1, 2) for p in data["img_points"]],
            axis=0)

        nx = max(1, img_w // bin_size)
        ny = max(1, img_h // bin_size)
        hist, _, _ = np.histogram2d(
            pts[:, 1], pts[:, 0], bins=[ny, nx],
            range=[[0, img_h], [0, img_w]])
        hist = cv2.GaussianBlur(hist, (0, 0), blur_sigma)
        density = cv2.resize(hist, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        norm = (density / density.max() * 255).astype(np.uint8) if density.max() > 0 else density.astype(np.uint8)
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        cv2.putText(heatmap, f"{npz.stem}  N={len(pts)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out_path = points_dir / f"{npz.stem}_density.png"
        cv2.imwrite(str(out_path), heatmap)
        print(f"  wrote {out_path} ({len(pts)} points)")


if __name__ == "__main__":
    process_videos(video_folder="../video/charuco_air", points_folder="../points/air", photos_folder="../photos/air", exclusive=False)
    save_density_heatmaps("../points/air")
    # process_videos(video_folder="../video/charuco_water", points_folder="../points/water", photos_folder="../photos/water", exclusive=False)
    # save_density_heatmaps("../points/water")
