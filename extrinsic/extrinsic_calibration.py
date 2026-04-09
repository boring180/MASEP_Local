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
        self.total_frames = int(data["total_frames"])

        self.camera_obj_points = {}
        self.camera_img_points = {}
        self.camera_frame_indices = {}
        for cam in self.camera_names:
            self.camera_obj_points[cam] = list(data[f"{cam}_obj_points"])
            self.camera_img_points[cam] = list(data[f"{cam}_img_points"])
            self.camera_frame_indices[cam] = data[f"{cam}_frame_indices"]

        self.camera_mtx = {}
        self.camera_dist = {}
        self.camera_extrinsic = {}
        self.camera_undistorted_img_points = {}

        print(f"Loaded {npz_path}")
        for cam in self.camera_names:
            print(f"  {cam}: {len(self.camera_obj_points[cam])} detections")

    def load_intrinsics(self, results_dir):
        """Load per-camera mtx/dist pickle files from *results_dir*."""
        for cam in self.camera_names:
            mtx_p = Path(results_dir) / f"mtx_{cam}.pkl"
            dist_p = Path(results_dir) / f"dist_{cam}.pkl"
            if mtx_p.exists() and dist_p.exists():
                with open(mtx_p, "rb") as f:
                    self.camera_mtx[cam] = pickle.load(f)
                with open(dist_p, "rb") as f:
                    self.camera_dist[cam] = pickle.load(f)
            else:
                print(f"  WARNING: intrinsics not found for {cam}")

    def _estimate_poses(self, use_undistorted=False):
        """Estimate board pose per camera per frame via solvePnP.

        If use_undistorted=True, uses pre-computed undistorted image points
        (from camera_undistorted_img_points) with zero distortion coefficients.
        """
        frame_poses = [{} for _ in range(self.total_frames)]
        for cam in self.camera_names:
            if cam not in self.camera_mtx:
                continue
            mtx = self.camera_mtx[cam]
            if use_undistorted and cam in self.camera_undistorted_img_points:
                img_pts = self.camera_undistorted_img_points[cam]
                dist = np.zeros(5)
            else:
                img_pts = self.camera_img_points[cam]
                dist = self.camera_dist[cam]
            for di, fi in enumerate(self.camera_frame_indices[cam]):
                obj = self.camera_obj_points[cam][di].astype(np.float32)
                img = img_pts[di].astype(np.float32)
                ret, rvec, tvec = cv2.solvePnP(obj, img, mtx, dist)
                if ret:
                    frame_poses[fi][cam] = (rvec, tvec, len(obj))
        return frame_poses

    def calibrate_extrinsic(self, center_camera, target_camera, weighted=False, use_undistorted=False):
        """Compute transformation from *target_camera* to *center_camera*."""
        frame_poses = self._estimate_poses(use_undistorted=use_undistorted)
        cam_Ts, ctr_Ts, ws = [], [], []
        for fp in frame_poses:
            if target_camera in fp and center_camera in fp:
                cam_Ts.append(fp[target_camera][1].flatten())
                ctr_Ts.append(fp[center_camera][1].flatten())
                ws.append(fp[target_camera][2] + fp[center_camera][2] if weighted else 1)

        if not cam_Ts:
            raise RuntimeError(f"No shared frames between {target_camera} and {center_camera}")

        cam_Ts, ctr_Ts = np.array(cam_Ts), np.array(ctr_Ts)
        ws = np.array(ws, dtype=float)
        print(f"{target_camera} -> {center_camera}: {len(cam_Ts)} shared frames")

        rot, trans, rmsd = self._point_cloud_matching(cam_Ts, ctr_Ts, ws)
        print(f"  RMSD: {rmsd:.6f}")

        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = trans
        self.camera_extrinsic[target_camera] = T
        return T

    def calibrate_all(self, center_camera, weighted=False, save_dir=None, use_undistorted=False):
        self.camera_extrinsic[center_camera] = np.eye(4)
        for cam in self.camera_names:
            if cam != center_camera:
                self.calibrate_extrinsic(center_camera, cam, weighted=weighted, use_undistorted=use_undistorted)
        if save_dir:
            self.save_extrinsics(save_dir)

    @staticmethod
    def _point_cloud_matching(P, Q, w):
        t = ((Q - P) * w[:, None]).sum(0) / w.sum()
        rot, rmsd = R.align_vectors(Q, P + t, weights=w)
        return rot, t, rmsd

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
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
