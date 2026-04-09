"""Pinhole OpenCV calibration + PyTorch distortion fit; full OpenCV calib as reference."""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

def opencv_full_calib(objpoints, imgpoints, image_size):
    # The filtered lists are used for the calibration
    objp_list, imgp_list = [], []
    for o, i in zip(objpoints, imgpoints):
        if len(i) >= 6:
            objp_list.append(np.asarray(o, np.float32))
            imgp_list.append(np.asarray(i, np.float32))

    # flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
    flags = 0
    rms_opencv, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objp_list, imgp_list, image_size, None, None, flags=flags
    )
    sse = n = 0
    for obj_pts, img_pts, rv, tv in zip(objp_list, imgp_list, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj_pts, rv, tv, mtx, dist)
        d = proj.reshape(-1, 2).astype(np.float64) - np.asarray(img_pts, np.float64).reshape(-1, 2)
        sse += float((d * d).sum())
        n += d.size // 2
    rmse = float(np.sqrt(sse / n))
    return {
        "K": mtx,
        "dist": dist.reshape(-1).astype(np.float64),
        "rvecs": rvecs,
        "tvecs": tvecs,
        "rmse_px": rmse,
        "filtered_obj": objp_list,
        "filtered_img": imgp_list,
        "rms_opencv": float(rms_opencv),
    }


class pytorch_distortion_fit():
    def __init__(self, objpoints, imgpoints, image_size, device='cpu', iteration_of_training=10, epoch_per_iteration=1000):
        self.image_size = image_size
            # The filtered lists are used for the calibration
        self.objp_list, self.imgp_list = [], []
        self.undistort_imgp_list = []
        for o, i in zip(objpoints, imgpoints):
            if len(i) >= 6:
                self.objp_list.append(np.asarray(o, np.float32))
                self.imgp_list.append(np.asarray(i, np.float32))
                self.undistort_imgp_list.append(np.asarray(i, np.float32))

        self.iteration_of_training = iteration_of_training
        self.epoch_per_iteration = epoch_per_iteration
        self.device = device
        self.loss_history = []

    def __str__(self):
        camera_matrix_msg = f"Camera Matrix: {self.mtx}"
        distortion_model_msg = f"Distortion Model: {self.model}"
        return f"{camera_matrix_msg}\n{distortion_model_msg}"

    def _calibrate_K_matrix(self):
        flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
        _, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objp_list, self.undistort_imgp_list, self.image_size, None, None, flags=flags)

    def _reproject_image_points(self):
        self.reprojected_imgp = []
        for objp, rvec, tvec in zip(self.objp_list, self.rvecs, self.tvecs):
            proj, _ = cv2.projectPoints(objp, rvec, tvec, self.mtx, self.dist)
            self.reprojected_imgp.append(proj.reshape(-1, 2))

    def reprojection_error(self):
        sse = 0
        n = 0
        for imgp, reproj in zip(self.undistort_imgp_list, self.reprojected_imgp):
            d = reproj.astype(np.float64) - imgp.reshape(-1, 2).astype(np.float64)
            sse += float((d * d).sum())
            n += d.size // 2
        return float(np.sqrt(sse / n))

    def _compute_normalized_coords(self, imgp_to_normalize):
        """Compute normalized camera coordinates and corresponding image points for all views."""
        all_normalized = []
        all_imgp = []
        for objp, imgp, rvec, tvec in zip(self.objp_list, imgp_to_normalize, self.rvecs, self.tvecs):
            R, _ = cv2.Rodrigues(rvec)
            cam_pts = (R @ objp.reshape(-1, 3).T + tvec).T
            x_n = cam_pts[:, 0] / cam_pts[:, 2]
            y_n = cam_pts[:, 1] / cam_pts[:, 2]
            all_normalized.append(np.stack([x_n, y_n], axis=1))
            all_imgp.append(imgp.reshape(-1, 2))
        return np.concatenate(all_normalized, axis=0), np.concatenate(all_imgp, axis=0)

    def _fit_pytorch_distortion(self):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        normalized, imgp = self._compute_normalized_coords(self.imgp_list)
        x_n_tensor = torch.tensor(normalized[:, 0], dtype=torch.float32, device=self.device)
        y_n_tensor = torch.tensor(normalized[:, 1], dtype=torch.float32, device=self.device)

        fx, fy = self.mtx[0, 0], self.mtx[1, 1]
        cx, cy = self.mtx[0, 2], self.mtx[1, 2]
        target_x = torch.tensor((imgp[:, 0] - cx) / fx, dtype=torch.float32, device=self.device)
        target_y = torch.tensor((imgp[:, 1] - cy) / fy, dtype=torch.float32, device=self.device)

        for epoch in range(self.epoch_per_iteration):
            optimizer.zero_grad()
            pred_x, pred_y = self.model(x_n_tensor, y_n_tensor)
            loss = criterion(pred_x, target_x) + criterion(pred_y, target_y)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch} loss: {loss.item()}, {self.model}")

    def compute_undistort_image_points(self, imgp_list=None):
        """Undistort image points by iteratively inverting the distortion model.

        The model maps undistorted_normalized → distorted_normalized. This function
        inverts that mapping: given distorted image points, finds the undistorted
        image points such that model(undistorted) ≈ distorted.

        Args:
            imgp_list: List of distorted image point arrays. If None, uses self.imgp_list.

        Returns:
            List of undistorted image point arrays, each shaped (N_i, 1, 2).
        """
        if imgp_list is None:
            imgp_list = self.imgp_list

        self.model.eval()
        fx, fy = self.mtx[0, 0], self.mtx[1, 1]
        cx, cy = self.mtx[0, 2], self.mtx[1, 2]

        all_pts, lengths = [], []
        for imgp in imgp_list:
            pts = imgp.reshape(-1, 2).astype(np.float64)
            all_pts.append(pts)
            lengths.append(len(pts))
        all_pts = np.concatenate(all_pts, axis=0)

        x_d = (all_pts[:, 0] - cx) / fx
        y_d = (all_pts[:, 1] - cy) / fy

        x_u, y_u = x_d.copy(), y_d.copy()
        with torch.no_grad():
            for _ in range(20):
                x_t = torch.tensor(x_u, dtype=torch.float32, device=self.device)
                y_t = torch.tensor(y_u, dtype=torch.float32, device=self.device)
                px, py = self.model(x_t, y_t)
                x_u += x_d - px.cpu().numpy()
                y_u += y_d - py.cpu().numpy()

        u = x_u * fx + cx
        v = y_u * fy + cy
        all_undist = np.stack([u, v], axis=1).astype(np.float32)

        result = []
        offset = 0
        for n in lengths:
            result.append(all_undist[offset:offset + n].reshape(-1, 1, 2))
            offset += n

        if imgp_list is self.imgp_list:
            self.undistort_imgp_list = result
        return result

    def fit(self):
        for i in range(self.iteration_of_training):
            self._calibrate_K_matrix()
            self._reproject_image_points()
            self._fit_pytorch_distortion()
            self.compute_undistort_image_points()
            print(f"Iteration {i}")
            print(f"Reprojection error: {self.reprojection_error()}")
        return self.mtx, self.dist, self.rvecs, self.tvecs

    def plot_loss(self, save_path="loss_plot.png", opencv_rmse=None):
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(self.loss_history, label="PyTorch distortion fit")
        ax1.set_xlabel("Global Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True)

        ax2.plot(self.loss_history, label="PyTorch distortion fit")
        ax2.set_yscale("log")
        ax2.set_xlabel("Global Step")
        ax2.set_ylabel("Loss (log scale)")
        ax2.set_title("Training Loss (log scale)")
        ax2.grid(True)

        epochs_per = self.epoch_per_iteration
        for ax in (ax1, ax2):
            for i in range(1, self.iteration_of_training):
                ax.axvline(x=i * epochs_per, color="red", linestyle="--", alpha=0.5)
            if opencv_rmse is not None:
                ax.axhline(y=opencv_rmse, color="green", linestyle="-", linewidth=2,
                           label=f"OpenCV RMSE ({opencv_rmse:.4f})")
            ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Loss plot saved to {save_path}")

    def plot_distortion_field(self, save_path="distortion_field.png", grid_n=20):
        import matplotlib.pyplot as plt
        w, h = self.image_size
        fx, fy = self.mtx[0, 0], self.mtx[1, 1]
        cx, cy = self.mtx[0, 2], self.mtx[1, 2]

        u = np.linspace(0, w, grid_n)
        v = np.linspace(0, h, grid_n)
        uu, vv = np.meshgrid(u, v)
        uu_flat = uu.flatten()
        vv_flat = vv.flatten()

        x_n = (uu_flat - cx) / fx
        y_n = (vv_flat - cy) / fy

        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x_n, dtype=torch.float32, device=self.device)
            y_t = torch.tensor(y_n, dtype=torch.float32, device=self.device)
            dx, dy = self.model(x_t, y_t)
            dx = dx.cpu().numpy()
            dy = dy.cpu().numpy()

        u_dist = dx * fx + cx
        v_dist = dy * fy + cy
        du = u_dist - uu_flat
        dv = v_dist - vv_flat

        arrow_scale = 50.0
        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(len(uu_flat)):
            ax.annotate("",
                        xy=(uu_flat[i] + du[i] * arrow_scale,
                            vv_flat[i] + dv[i] * arrow_scale),
                        xytext=(uu_flat[i], vv_flat[i]),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        ax.set_title(f"Distortion Vector Field ({self.model.__class__.__name__}, arrows scaled {arrow_scale:.0f}x)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Distortion field saved to {save_path}")

    def plot_reprojection_error_distribution(self, save_path="reprojection_error_distribution.png"):
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        all_sq_dists = []
        for imgp, reproj in zip(self.undistort_imgp_list, self.reprojected_imgp):
            d = reproj.astype(np.float64) - imgp.reshape(-1, 2).astype(np.float64)
            sq_dist = np.sum(d ** 2, axis=1)
            all_sq_dists.append(sq_dist)
        all_sq_dists = np.concatenate(all_sq_dists)

        p95 = np.percentile(all_sq_dists, 95)
        clipped = all_sq_dists[all_sq_dists <= p95]
        n_outliers = len(all_sq_dists) - len(clipped)

        kde = gaussian_kde(clipped)
        x = np.linspace(0, p95, 500)
        density = kde(x)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(x, density, alpha=0.3, color="steelblue")
        ax.plot(x, density, color="steelblue", linewidth=2)
        ax.axvline(np.mean(all_sq_dists), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean = {np.mean(all_sq_dists):.4f} px²")
        ax.axvline(np.median(all_sq_dists), color="orange", linestyle="--", linewidth=1.5,
                   label=f"Median = {np.median(all_sq_dists):.4f} px²")
        ax.set_xlim(0, p95 * 1.05)
        ax.set_xlabel("Squared Reprojection Distance (px²)")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"PDF of Per-Point Squared Reprojection Error "
                     f"(95th percentile, {n_outliers} outliers clipped)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Reprojection error distribution saved to {save_path}")
        print(f"  Total points: {len(all_sq_dists)}, "
              f"Mean: {np.mean(all_sq_dists):.4f} px², "
              f"Median: {np.median(all_sq_dists):.4f} px², "
              f"95th pct: {p95:.4f} px², "
              f"Max: {np.max(all_sq_dists):.4f} px²")

    def ordinary_polynomial_distortion(self):
        class OrdinaryPolynomialDistortion(nn.Module):
            def __init__(self):
                super().__init__()
                self.k1 = nn.Parameter(torch.tensor(0.0))
                self.k2 = nn.Parameter(torch.tensor(0.0))
                self.k3 = nn.Parameter(torch.tensor(0.0))
                self.p1 = nn.Parameter(torch.tensor(0.0))
                self.p2 = nn.Parameter(torch.tensor(0.0))

            def forward(self, x, y):
                r_squared = x**2 + y**2
                x = x * ( 1 + self.k1 * r_squared + self.k2 * r_squared ** 2 + self.k3 * r_squared ** 3)
                y = y * ( 1 + self.k1 * r_squared + self.k2 * r_squared ** 2 + self.k3 * r_squared ** 3)

                x = x + 2 * self.p1 * x * y + self.p2 * (r_squared + 2 * x**2)
                y = y + 2 * self.p1 * x * y + self.p2 * (r_squared + 2 * y**2)
                return x, y

            def __str__(self):
                return f"{self.__class__.__name__}(k1={self.k1.item():.4f}, k2={self.k2.item():.4f}, k3={self.k3.item():.4f}, p1={self.p1.item():.4f}, p2={self.p2.item():.4f})"

        self.model = OrdinaryPolynomialDistortion().to(self.device)

    def mlp_distortion(self):
        class MLPDistortion(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(2, 10),
                    nn.ReLU(),
                    nn.Linear(10, 10),
                    nn.ReLU(),
                    nn.Linear(10, 2),
                )
            def forward(self, x, y):
                out = self.mlp(torch.stack([x, y], dim=1))
                return out[:, 0], out[:, 1]
            def __str__(self):
                return f"{self.__class__.__name__}(mlp={self.mlp})"

        self.model = MLPDistortion().to(self.device)