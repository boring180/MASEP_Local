"""Extrinsic calibration from multi-camera ChArUco point data (.npz)."""

import cv2
import numpy as np
import pickle
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


class ExtrinsicCalibration:
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.camera_names = list(data["camera_names"])
        self.img_size = tuple(data["img_size"])
        self.square_size = float(data["square_size"])
        self.marker_size = float(data["marker_size"])
        self.n_frames = len(data[f"{self.camera_names[0]}_obj_points"])

        self.camera_obj_points = {}
        self.camera_img_points = {}
        for cam in self.camera_names:
            self.camera_obj_points[cam] = list(data[f"{cam}_obj_points"])
            self.camera_img_points[cam] = list(data[f"{cam}_img_points"])

        self.camera_mtx = {}
        self.camera_dist = {}
        self.camera_extrinsic = {}

        print(f"Loaded {npz_path}: {self.n_frames} synced frames")
        for cam in self.camera_names:
            n_valid = sum(1 for p in self.camera_obj_points[cam] if p is not None)
            print(f"  {cam}: {n_valid}/{self.n_frames} detected")

    def load_intrinsics(self, calib_dir):
        """Load per-camera intrinsics from opencv .npz files in calib_dir."""
        for cam in self.camera_names:
            p = Path(calib_dir) / f"{cam}_opencv.npz"
            if p.exists():
                d = np.load(str(p))
                self.camera_mtx[cam] = d["K"]
                self.camera_dist[cam] = d["dist"]
            else:
                print(f"  WARNING: intrinsics not found for {cam}")

    def _estimate_poses(self, min_pts=6, max_reproj_err=3.0):
        """Estimate board pose per camera per synced frame via solvePnP.

        Rejects poses with non-collinear failures, board behind the camera,
        or mean reprojection error above `max_reproj_err` pixels.
        """
        frame_poses = []
        for fi in range(self.n_frames):
            poses = {}
            for cam in self.camera_names:
                if cam not in self.camera_mtx:
                    continue
                obj = self.camera_obj_points[cam][fi]
                img = self.camera_img_points[cam][fi]
                if obj is None or img is None or len(obj) < min_pts:
                    continue
                xy = np.asarray(obj, np.float64)[:, :2]
                s = np.linalg.svd(xy - xy.mean(0), compute_uv=False)
                if s[1] < 1e-6 * s[0]:
                    continue
                ret, rvec, tvec = cv2.solvePnP(
                    obj.astype(np.float32), img.astype(np.float32),
                    self.camera_mtx[cam], self.camera_dist[cam], flags=cv2.SOLVEPNP_IPPE)
                if not ret or float(tvec[2, 0]) <= 0:
                    continue
                proj, _ = cv2.projectPoints(obj.astype(np.float32), rvec, tvec,
                                            self.camera_mtx[cam], self.camera_dist[cam])
                d = proj.reshape(-1, 2).astype(np.float64) - img.reshape(-1, 2).astype(np.float64)
                err = float(np.sqrt((d * d).sum(1)).mean())
                if err > max_reproj_err:
                    continue
                poses[cam] = (rvec, tvec, len(obj))
            frame_poses.append(poses)
        return frame_poses

    @staticmethod
    def _pose_to_T(rvec, tvec):
        """Convert solvePnP output to 4x4 transform (camera ← board)."""
        T = np.eye(4)
        T[:3, :3], _ = cv2.Rodrigues(rvec)
        T[:3, 3] = tvec.flatten()
        return T

    def calibrate_extrinsic(self, center_camera, target_camera):
        """Compute T_center_target = T_center_board @ inv(T_target_board), averaged over shared frames."""
        frame_poses = self._estimate_poses()
        Ts = []
        for fp in frame_poses:
            if target_camera in fp and center_camera in fp:
                T_ctr_board = self._pose_to_T(*fp[center_camera][:2])
                T_tgt_board = self._pose_to_T(*fp[target_camera][:2])
                T_ctr_tgt = T_ctr_board @ np.linalg.inv(T_tgt_board)
                Ts.append(T_ctr_tgt)

        if not Ts:
            raise RuntimeError(f"No shared frames between {target_camera} and {center_camera}")

        # Quaternion-based rotation mean — robust to the sign ambiguity and
        # sidesteps scipy's eigh() which can fail to converge on degenerate sets.
        quats = R.from_matrix([T[:3, :3] for T in Ts]).as_quat()  # (N, 4) xyzw
        quats *= np.sign(quats @ quats[0])[:, None]  # hemisphere-align to first
        q_mean = quats.mean(axis=0)
        q_mean /= np.linalg.norm(q_mean)
        avg_rot = R.from_quat(q_mean)
        avg_trans = np.mean([T[:3, 3] for T in Ts], axis=0)

        T = np.eye(4)
        T[:3, :3] = avg_rot.as_matrix()
        T[:3, 3] = avg_trans
        self.camera_extrinsic[target_camera] = T

        # Validation loss: apply the averaged extrinsic and measure per-frame residual
        # between the predicted center-camera board pose and the directly-observed one.
        # Translation residual is in the center-camera frame; rotation residual is the
        # axis-angle error of R_pred^T @ R_obs expressed as xyz Euler (degrees).
        t_res, r_res = [], []
        for fp in frame_poses:
            if target_camera not in fp or center_camera not in fp:
                continue
            T_ctr_board_obs = self._pose_to_T(*fp[center_camera][:2])
            T_tgt_board = self._pose_to_T(*fp[target_camera][:2])
            T_ctr_board_pred = T @ T_tgt_board
            t_res.append(T_ctr_board_obs[:3, 3] - T_ctr_board_pred[:3, 3])
            R_err = T_ctr_board_pred[:3, :3].T @ T_ctr_board_obs[:3, :3]
            r_res.append(R.from_matrix(R_err).as_euler("xyz", degrees=True))
        t_res = np.asarray(t_res)
        r_res = np.asarray(r_res)

        print(f"{target_camera} -> {center_camera}: {len(Ts)} shared frames")
        print(f"  translation residual mean (x,y,z): {t_res.mean(axis=0)}")
        print(f"  translation residual std  (x,y,z): {t_res.std(axis=0)}")
        print(f"  rotation residual mean (rx,ry,rz) deg: {r_res.mean(axis=0)}")
        print(f"  rotation residual std  (rx,ry,rz) deg: {r_res.std(axis=0)}")
        print(f"  T:\n{T}")
        return T

    def calibrate_all(self, center_camera, save_dir=None):
        self.camera_extrinsic[center_camera] = np.eye(4)
        for cam in self.camera_names:
            if cam != center_camera:
                self.calibrate_extrinsic(center_camera, cam)
        if save_dir:
            self.save_extrinsics(save_dir)

    def save_extrinsics(self, save_dir):
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        for cam, T in self.camera_extrinsic.items():
            with open(d / f"extrinsic_{cam}.pkl", "wb") as f:
                pickle.dump(T, f)

    def load_extrinsics(self, load_dir):
        d = Path(load_dir)
        for cam in self.camera_names:
            p = d / f"extrinsic_{cam}.pkl"
            if p.exists():
                with open(p, "rb") as f:
                    self.camera_extrinsic[cam] = pickle.load(f)

    def visualize(self, save_path=None):
        fig = plt.figure(figsize=(15, 10))
        # Collect all positions to compute equal axis limits
        positions = []
        for cam in self.camera_names:
            if cam in self.camera_extrinsic:
                positions.append(self.camera_extrinsic[cam][:3, 3])
        positions = np.array(positions)
        center = positions.mean(axis=0)
        half_range = np.abs(positions - center).max() * 1.5

        for idx, (elev, azim, roll) in enumerate([(45, 0, 0), (45, 45, 0), (-90, 90, 0)], 1):
            ax = fig.add_subplot(1, 3, idx, projection="3d", elev=elev, azim=azim, roll=roll)
            ax.set_box_aspect([1, 1, 1])
            axes_d = np.eye(3) * self.square_size
            for cam in self.camera_names:
                if cam not in self.camera_extrinsic:
                    continue
                T = self.camera_extrinsic[cam]
                pos = T[:3, 3]
                for i, c in enumerate(["red", "green", "blue"]):
                    d = T[:3, :3] @ axes_d[i]
                    ax.quiver(*pos, *d, color=c, length=1)
                    if i == 0:
                        ax.text(*pos, cam)
            ax.set_xlim(center[0] - half_range, center[0] + half_range)
            ax.set_ylim(center[1] - half_range, center[1] + half_range)
            ax.set_zlim(center[2] - half_range, center[2] + half_range)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


if __name__ == "__main__":
    ec = ExtrinsicCalibration("../points/air/extrinsic.npz")
    ec.load_intrinsics("../calibration/air")
    ec.calibrate_all(center_camera="cam1", save_dir="../calibration/air")
    ec.visualize(save_path="extrinsic_air.png")
