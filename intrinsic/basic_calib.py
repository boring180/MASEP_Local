"""Bare-minimum OpenCV calibration — no filtering, no refit. Sanity check."""

import numpy as np
import cv2
from pathlib import Path


def basic_calibrate(points_file):
    data = np.load(points_file, allow_pickle=True)
    obj_points = [np.asarray(a, np.float32) for a in data["obj_points"]]
    img_points = [np.asarray(a, np.float32) for a in data["img_points"]]
    img_size = tuple(int(x) for x in data["img_size"])

    # OpenCV needs >=4 non-collinear points per view. Nothing else filtered.
    pairs = []
    for o, i in zip(obj_points, img_points):
        if len(i) < 4:
            continue
        xy = np.asarray(o, np.float64)[:, :2]
        s = np.linalg.svd(xy - xy.mean(0), compute_uv=False)
        if s[1] < 1e-6 * s[0]:
            continue
        pairs.append((o, i))
    obj_points = [p[0] for p in pairs]
    img_points = [p[1] for p in pairs]

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None)

    print(f"\n{Path(points_file).stem}  ({len(obj_points)} views, img_size={img_size})")
    print(f"  RMS (OpenCV) = {rms:.4f} px")
    print(f"  K =\n{K}")
    print(f"  fx/fy = {K[0,0]/K[1,1]:.4f}   cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print(f"  dist = {dist.ravel()}")
    return K, dist, rms


if __name__ == "__main__":
    for npz in sorted(Path("../points/air").glob("cam*.npz")):
        basic_calibrate(str(npz))
