"""Compare charuco localization under water using the three underwater
intrinsic methods: air_direct, water_opencv, and refractive."""

from localization import (
    METHODS,
    load_extrinsics, load_intrinsics,
    make_charuco_detector, detect_video, localize_from_detections,
)
from result_analysis import (
    extract_positions, filter_best_frames,
    compare_methods, plot_inter_camera_mse, plot_scatter_grid,
)

INTRINSIC_DIR = "../calibration/water"
EXTRINSIC_DIR = "../calibration/air"  # extrinsics were calibrated in air
VIDEO_DIR = "../video/charuco_water"
VIDEOS = [
    "charuco_underwater1.mp4",
    "charuco_underwater2.mp4",
    "charuco_underwater3.mp4",
]


def main():
    board, detector = make_charuco_detector()
    extrinsics = load_extrinsics(EXTRINSIC_DIR)

    intrinsics = {m: load_intrinsics(INTRINSIC_DIR, m) for m in METHODS}

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

    # Keep only the best 95% of frames (drop worst 5% by inter-camera spread)
    # per method, so stats/plots aren't dragged by occasional PnP outliers.
    all_results = {m: filter_best_frames(r, keep=0.95) for m, r in all_results.items()}

    all_positions = {}
    for method in METHODS:
        all_positions[method] = extract_positions(all_results[method])
        n = sum(1 for p in all_positions[method] if p is not None)
        print(f"'{method}': {n}/{len(all_positions[method])} frames localized (top 95%)")

    compare_methods(all_positions)
    plot_inter_camera_mse(all_results)
    # plot_scatter_grid(all_results)


if __name__ == "__main__":
    main()
