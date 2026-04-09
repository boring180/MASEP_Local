"""Pipeline comparison: OpenCV calibration vs Proposed (PyTorch distortion fit).

Compares intrinsic + extrinsic calibration with train/test split.
- Method 1 (OpenCV): calibrateCamera → K, dist; solvePnP with dist
- Method 2 (Proposed): PyTorch distortion fit → K, model; undistort points then solvePnP with zero dist
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extrinsic"))

from intrinsic_calibration import opencv_full_calib, pytorch_distortion_fit
from extrinsic_calibration import ExtrinsicCalibration


# ── helpers ──────────────────────────────────────────────────────────────

def split_indices(n, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(n * test_ratio))
    return idx[n_test:], idx[:n_test]  # train, test


def select_by_idx(obj_list, img_list, indices):
    return [obj_list[i] for i in indices], [img_list[i] for i in indices]


def reproj_error_pnp(obj_list, img_list, K, dist):
    """Per-point reprojection error: solvePnP then reproject."""
    errors = []
    for obj, img in zip(obj_list, img_list):
        obj = np.asarray(obj, np.float32)
        img = np.asarray(img, np.float32).reshape(-1, 2)
        if len(obj) < 4:
            continue
        ret, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
        if ret:
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            d = proj.reshape(-1, 2) - img
            errors.extend(np.sqrt((d ** 2).sum(axis=1)))
    return np.array(errors)


def extrinsic_pnp(obj_list, img_list, frame_indices, K, dist, total_frames):
    """Run solvePnP per frame, return frame_poses dict-list."""
    frame_poses = [{} for _ in range(total_frames)]
    for di, fi in enumerate(frame_indices):
        obj = np.asarray(obj_list[di], np.float32)
        img = np.asarray(img_list[di], np.float32).reshape(-1, 2)
        if len(obj) < 4:
            continue
        ret, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
        if ret:
            frame_poses[fi] = (rvec, tvec, len(obj))
    return frame_poses


def point_cloud_matching(P, Q, w):
    """Same as ExtrinsicCalibration._point_cloud_matching."""
    return ExtrinsicCalibration._point_cloud_matching(P, Q, w)


def compute_extrinsic(center_poses, target_poses, total_frames):
    """Compute extrinsic T (target→center) from per-frame poses."""
    cam_Ts, ctr_Ts, ws = [], [], []
    for fi in range(total_frames):
        if center_poses[fi] and target_poses[fi]:
            ctr_Ts.append(center_poses[fi][1].flatten())
            cam_Ts.append(target_poses[fi][1].flatten())
            ws.append(1)
    if not cam_Ts:
        return None, 0, 0.0
    cam_Ts, ctr_Ts = np.array(cam_Ts), np.array(ctr_Ts)
    ws = np.array(ws, dtype=float)
    rot, trans, rmsd = point_cloud_matching(cam_Ts, ctr_Ts, ws)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = trans
    return T, len(cam_Ts), rmsd


def cross_camera_reproj_error(center_poses, target_obj, target_img, target_fi,
                               extrinsic_T, K_target, dist_target, total_frames):
    """Cross-camera reprojection: center cam pose → project into target cam."""
    T_inv = np.linalg.inv(extrinsic_T)
    target_by_frame = {}
    for di, fi in enumerate(target_fi):
        target_by_frame[fi] = (target_obj[di], target_img[di])

    errors = []
    for fi in range(total_frames):
        if not center_poses[fi] or fi not in target_by_frame:
            continue
        rvec_c, tvec_c, _ = center_poses[fi]
        R_c, _ = cv2.Rodrigues(rvec_c)
        R_t = T_inv[:3, :3] @ R_c
        t_t = T_inv[:3, :3] @ tvec_c.flatten() + T_inv[:3, 3]
        rvec_t, _ = cv2.Rodrigues(R_t)

        t_obj, t_img = target_by_frame[fi]
        t_obj = np.asarray(t_obj, np.float32)
        t_img = np.asarray(t_img, np.float32).reshape(-1, 2)
        proj, _ = cv2.projectPoints(t_obj, rvec_t, t_t.reshape(3, 1), K_target, dist_target)
        d = proj.reshape(-1, 2) - t_img
        errors.extend(np.sqrt((d ** 2).sum(axis=1)))
    return np.array(errors)


# ── main ─────────────────────────────────────────────────────────────────

def run_single_camera(points_file, test_ratio=0.2, seed=42):
    """Compare intrinsic calibration methods on single-camera data."""
    data = np.load(points_file, allow_pickle=True)
    obj_all = [arr.astype(np.float32) for arr in data["obj_points"]]
    img_all = [arr.astype(np.float32) for arr in data["img_points"]]
    img_size = tuple(int(x) for x in data["img_size"])
    n = len(obj_all)

    train_idx, test_idx = split_indices(n, test_ratio, seed)
    obj_train, img_train = select_by_idx(obj_all, img_all, train_idx)
    obj_test, img_test = select_by_idx(obj_all, img_all, test_idx)
    print(f"Single-camera: {n} frames, train={len(train_idx)}, test={len(test_idx)}")

    # ── Method 1: OpenCV ──
    print("\n" + "=" * 60)
    print("Method 1: OpenCV full calibration")
    print("=" * 60)
    cv_result = opencv_full_calib(obj_train, img_train, img_size)
    K_cv, dist_cv = cv_result["K"], cv_result["dist"]
    print(f"  Train RMSE: {cv_result['rmse_px']:.4f} px")

    err_cv_test = reproj_error_pnp(obj_test, img_test, K_cv, dist_cv)
    print(f"  Test  RMSE: {np.sqrt(np.mean(err_cv_test ** 2)):.4f} px  (n={len(err_cv_test)})")

    # ── Method 2: Proposed ──
    print("\n" + "=" * 60)
    print("Method 2: Proposed (PyTorch distortion fit)")
    print("=" * 60)
    pt_fit = pytorch_distortion_fit(obj_train, img_train, img_size)
    pt_fit.ordinary_polynomial_distortion()
    mtx_pt, _, _, _ = pt_fit.fit()
    print(f"  Train RMSE: {pt_fit.reprojection_error():.4f} px")

    # Undistort test image points, then PnP with zero dist
    undist_test = pt_fit.compute_undistort_image_points(img_test)
    err_pt_test = reproj_error_pnp(obj_test, undist_test, mtx_pt, np.zeros(5))
    print(f"  Test  RMSE: {np.sqrt(np.mean(err_pt_test ** 2)):.4f} px  (n={len(err_pt_test)})")

    # ── Plots ──
    pt_fit.plot_loss(save_path="loss_plot.png", opencv_rmse=cv_result["rmse_px"])
    pt_fit.plot_distortion_field(save_path="distortion_field.png")
    pt_fit.plot_reprojection_error_distribution(save_path="reprojection_error_distribution.png")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Summary (single-camera intrinsic)")
    print("=" * 60)
    print(f"  {'Method':<30} {'Train RMSE':>12} {'Test RMSE':>12}")
    print(f"  {'OpenCV':<30} {cv_result['rmse_px']:>12.4f} {np.sqrt(np.mean(err_cv_test**2)):>12.4f}")
    print(f"  {'Proposed (poly distortion)':<30} {pt_fit.reprojection_error():>12.4f} {np.sqrt(np.mean(err_pt_test**2)):>12.4f}")

    return {
        "opencv": {"K": K_cv, "dist": dist_cv},
        "proposed": {"K": mtx_pt, "model": pt_fit},
    }


def run_multi_camera(npz_path, center_camera=None, test_ratio=0.2, seed=42):
    """Compare full intrinsic + extrinsic pipeline on multi-camera data."""
    data = np.load(npz_path, allow_pickle=True)
    camera_names = list(data["camera_names"])
    img_size = tuple(data["img_size"])
    total_frames = int(data["total_frames"])

    if center_camera is None:
        center_camera = camera_names[0]

    # Load per-camera data
    cam_data = {}
    for cam in camera_names:
        cam_data[cam] = {
            "obj": list(data[f"{cam}_obj_points"]),
            "img": list(data[f"{cam}_img_points"]),
            "fi": data[f"{cam}_frame_indices"],
        }

    # Train/test split by frame index
    all_frames = np.arange(total_frames)
    train_frames_idx, test_frames_idx = split_indices(total_frames, test_ratio, seed)
    train_frames = set(train_frames_idx)
    test_frames = set(test_frames_idx)
    print(f"Multi-camera: {len(camera_names)} cameras, {total_frames} frames")
    print(f"  Train frames: {len(train_frames)}, Test frames: {len(test_frames)}")

    # Split per-camera data by frame membership
    train_data, test_data = {}, {}
    for cam in camera_names:
        tr_obj, tr_img, tr_fi = [], [], []
        te_obj, te_img, te_fi = [], [], []
        for di, fi in enumerate(cam_data[cam]["fi"]):
            if fi in train_frames:
                tr_obj.append(cam_data[cam]["obj"][di])
                tr_img.append(cam_data[cam]["img"][di])
                tr_fi.append(fi)
            elif fi in test_frames:
                te_obj.append(cam_data[cam]["obj"][di])
                te_img.append(cam_data[cam]["img"][di])
                te_fi.append(fi)
        train_data[cam] = {"obj": tr_obj, "img": tr_img, "fi": np.array(tr_fi)}
        test_data[cam] = {"obj": te_obj, "img": te_img, "fi": np.array(te_fi)}
        print(f"  {cam}: {len(tr_obj)} train, {len(te_obj)} test detections")

    # ── Method 1: OpenCV ──
    print("\n" + "=" * 60)
    print("Method 1: OpenCV full calibration")
    print("=" * 60)
    cv_intrinsics = {}
    for cam in camera_names:
        result = opencv_full_calib(train_data[cam]["obj"], train_data[cam]["img"], img_size)
        cv_intrinsics[cam] = {"K": result["K"], "dist": result["dist"]}
        err_test = reproj_error_pnp(test_data[cam]["obj"], test_data[cam]["img"],
                                     result["K"], result["dist"])
        print(f"  {cam}: train RMSE={result['rmse_px']:.4f}, "
              f"test RMSE={np.sqrt(np.mean(err_test**2)):.4f} ({len(err_test)} pts)")

    # Extrinsic (OpenCV)
    print("\nOpenCV extrinsic (train):")
    cv_train_poses = {}
    for cam in camera_names:
        cv_train_poses[cam] = extrinsic_pnp(
            train_data[cam]["obj"], train_data[cam]["img"], train_data[cam]["fi"],
            cv_intrinsics[cam]["K"], cv_intrinsics[cam]["dist"], total_frames)

    cv_extrinsics = {}
    cv_extrinsics[center_camera] = np.eye(4)
    for cam in camera_names:
        if cam == center_camera:
            continue
        T, n_shared, rmsd = compute_extrinsic(cv_train_poses[center_camera],
                                               cv_train_poses[cam], total_frames)
        cv_extrinsics[cam] = T
        print(f"  {cam} -> {center_camera}: {n_shared} shared, RMSD={rmsd:.6f}")

    # Cross-camera test reprojection (OpenCV)
    print("\nOpenCV cross-camera reprojection (test):")
    cv_test_poses = {}
    for cam in camera_names:
        cv_test_poses[cam] = extrinsic_pnp(
            test_data[cam]["obj"], test_data[cam]["img"], test_data[cam]["fi"],
            cv_intrinsics[cam]["K"], cv_intrinsics[cam]["dist"], total_frames)

    for cam in camera_names:
        if cam == center_camera or cv_extrinsics.get(cam) is None:
            continue
        err = cross_camera_reproj_error(
            cv_test_poses[center_camera],
            test_data[cam]["obj"], test_data[cam]["img"], test_data[cam]["fi"],
            cv_extrinsics[cam], cv_intrinsics[cam]["K"], cv_intrinsics[cam]["dist"],
            total_frames)
        if len(err) > 0:
            print(f"  {center_camera}->{cam}: RMSE={np.sqrt(np.mean(err**2)):.4f} ({len(err)} pts)")

    # ── Method 2: Proposed ──
    print("\n" + "=" * 60)
    print("Method 2: Proposed (PyTorch distortion fit)")
    print("=" * 60)
    pt_intrinsics = {}
    pt_models = {}
    for cam in camera_names:
        pt_fit = pytorch_distortion_fit(train_data[cam]["obj"], train_data[cam]["img"], img_size)
        pt_fit.ordinary_polynomial_distortion()
        mtx, _, _, _ = pt_fit.fit()
        pt_intrinsics[cam] = {"K": mtx.copy()}
        pt_models[cam] = pt_fit

        undist_test = pt_fit.compute_undistort_image_points(test_data[cam]["img"])
        err_test = reproj_error_pnp(test_data[cam]["obj"], undist_test, mtx, np.zeros(5))
        print(f"  {cam}: train RMSE={pt_fit.reprojection_error():.4f}, "
              f"test RMSE={np.sqrt(np.mean(err_test**2)):.4f} ({len(err_test)} pts)")

    # Extrinsic (Proposed): undistort train points, PnP with zero dist
    print("\nProposed extrinsic (train):")
    pt_train_poses = {}
    for cam in camera_names:
        undist_train = pt_models[cam].compute_undistort_image_points(train_data[cam]["img"])
        pt_train_poses[cam] = extrinsic_pnp(
            train_data[cam]["obj"], undist_train, train_data[cam]["fi"],
            pt_intrinsics[cam]["K"], np.zeros(5), total_frames)

    pt_extrinsics = {}
    pt_extrinsics[center_camera] = np.eye(4)
    for cam in camera_names:
        if cam == center_camera:
            continue
        T, n_shared, rmsd = compute_extrinsic(pt_train_poses[center_camera],
                                               pt_train_poses[cam], total_frames)
        pt_extrinsics[cam] = T
        print(f"  {cam} -> {center_camera}: {n_shared} shared, RMSD={rmsd:.6f}")

    # Cross-camera test reprojection (Proposed)
    print("\nProposed cross-camera reprojection (test):")
    pt_test_poses = {}
    for cam in camera_names:
        undist_test = pt_models[cam].compute_undistort_image_points(test_data[cam]["img"])
        pt_test_poses[cam] = extrinsic_pnp(
            test_data[cam]["obj"], undist_test, test_data[cam]["fi"],
            pt_intrinsics[cam]["K"], np.zeros(5), total_frames)

    for cam in camera_names:
        if cam == center_camera or pt_extrinsics.get(cam) is None:
            continue
        undist_test_img = pt_models[cam].compute_undistort_image_points(test_data[cam]["img"])
        err = cross_camera_reproj_error(
            pt_test_poses[center_camera],
            test_data[cam]["obj"], undist_test_img, test_data[cam]["fi"],
            pt_extrinsics[cam], pt_intrinsics[cam]["K"], np.zeros(5),
            total_frames)
        if len(err) > 0:
            print(f"  {center_camera}->{cam}: RMSE={np.sqrt(np.mean(err**2)):.4f} ({len(err)} pts)")


def main():
    single_cam_npz = "calibration_points_charuco.npz"
    multi_cam_npz = None  # set to multi-camera npz path when available

    if Path(single_cam_npz).exists():
        run_single_camera(single_cam_npz)

    if multi_cam_npz and Path(multi_cam_npz).exists():
        run_multi_camera(multi_cam_npz)


if __name__ == "__main__":
    main()
