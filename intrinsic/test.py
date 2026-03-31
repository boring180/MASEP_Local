"""Run intrinsic calibration from a .npz point file."""

import numpy as np
from intrinsic_calibration import opencv_full_calib, pytorch_distortion_fit


def main():
    points_file = "calibration_points_charuco.npz"
    data = np.load(points_file, allow_pickle=True)
    obj_points_list = [arr.astype(np.float32) for arr in data["obj_points"]]
    img_points_list = [arr.astype(np.float32) for arr in data["img_points"]]
    img_size = tuple(int(x) for x in data["img_size"])

    opencv_result = opencv_full_calib(obj_points_list, img_points_list, img_size)

    pt_fit = pytorch_distortion_fit(obj_points_list, img_points_list, img_size)
    pt_fit.ordinary_polynomial_distortion()
    pt_fit.fit()

    print(f"OpenCV K:\n{opencv_result['K']}")
    print(f"OpenCV dist: {opencv_result['dist']}")
    print(f"OpenCV RMSE: {opencv_result['rmse_px']:.6f}")
    print(f"PyTorch fit: {pt_fit}")
    print(f"PyTorch RMSE: {pt_fit.reprojection_error():.6f}")

    pt_fit.plot_loss(opencv_rmse=opencv_result["rmse_px"])
    pt_fit.plot_distortion_field()
    pt_fit.plot_reprojection_error_distribution()


if __name__ == "__main__":
    main()
