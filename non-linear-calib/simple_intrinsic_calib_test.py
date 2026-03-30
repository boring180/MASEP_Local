import numpy as np
import cv2

from simple_intrinsic_calib import (
    opencv_full_calib,
    pytorch_distortion_fit
)

def main():
    points_file = "calibration_points_charuco.npz"
    data = np.load(points_file, allow_pickle=True)
    obj_points_list = [arr.astype(np.float32) for arr in data["obj_points"]]
    img_points_list = [arr.astype(np.float32) for arr in data["img_points"]]
    img_size = tuple(int(x) for x in data["img_size"])

    opencv_result = opencv_full_calib(obj_points_list, img_points_list, img_size)
    pytorch_distortion_fit_result = pytorch_distortion_fit(
        obj_points_list, img_points_list, img_size
    )
    print(f"OpenCV K: {opencv_result['K']}")
    print(f"OpenCV dist: {opencv_result['dist']}")
    return
    pytorch_distortion_fit_result.ordinary_polynomial_distortion()
    # pytorch_distortion_fit_result.mlp_distortion()
    pytorch_distortion_fit_result.fit()
    print(f"OpenCV K: {opencv_result['K']}")
    print(f"OpenCV dist: {opencv_result['dist']}")
    print(f"PyTorch distortion fit: {pytorch_distortion_fit_result}")
    print(f"OpenCV reprojection error: {opencv_result['rmse_px']}")
    print(f"PyTorch distortion fit reprojection error: {pytorch_distortion_fit_result.reprojection_error()}")
    pytorch_distortion_fit_result.plot_loss(opencv_rmse=opencv_result['rmse_px'])
    pytorch_distortion_fit_result.plot_distortion_field()
    pytorch_distortion_fit_result.plot_per_view_reprojection_error()

if __name__ == "__main__":
    main()