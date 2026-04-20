"""Compare charuco localization under water using 3 intrinsic methods."""

from localization import (
    load_extrinsics, load_opencv_intrinsics, load_pytorch_intrinsics,
    make_charuco_detector, detect_video, localize_from_detections,
)
from result_analysis import (
    extract_positions, compare_methods, plot_inter_camera_mse, plot_scatter_grid,
)

CALIB_DIR = "../calibration/water"
VIDEO_DIR = "../video/charuco_water"
VIDEOS = [
    "charuco_underwater1.mp4",
    "charuco_underwater2.mp4",
    "charuco_underwater3.mp4",
]
METHODS = ["opencv", "poly", "mlp"]


def main():
    board, detector = make_charuco_detector()
    extrinsics = load_extrinsics(CALIB_DIR)

    intrinsics = {
        "opencv": load_opencv_intrinsics(CALIB_DIR),
        "poly": load_pytorch_intrinsics(CALIB_DIR, "poly"),
        "mlp": load_pytorch_intrinsics(CALIB_DIR, "mlp"),
    }

    all_results = {m: [] for m in METHODS}
    for video in VIDEOS:
        video_path = f"{VIDEO_DIR}/{video}"
        print(f"\nDetecting: {video} ...")
        detections = detect_video(video_path, board, detector)
        print(f"  {sum(1 for d in detections if d)}/{len(detections)} frames with detections")

        for method in METHODS:
            print(f"  Localizing with '{method}' ...")
            results = [localize_from_detections(det, intrinsics[method], extrinsics, method)
                       for det in detections]
            all_results[method].extend(results)

    all_positions = {}
    for method in METHODS:
        all_positions[method] = extract_positions(all_results[method])
        n = sum(1 for p in all_positions[method] if p is not None)
        print(f"'{method}': {n}/{len(all_positions[method])} frames localized")

    compare_methods(all_positions)
    plot_inter_camera_mse(all_results)
    plot_scatter_grid(all_results)


if __name__ == "__main__":
    main()
