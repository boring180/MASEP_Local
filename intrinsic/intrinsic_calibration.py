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
    def __init__(self, objpoints, imgpoints, image_size, device='cpu', epochs=10000, val_ratio=0.2):
        self.image_size = image_size
        self.objp_list, self.imgp_list = [], []
        for o, i in zip(objpoints, imgpoints):
            if len(i) >= 6:
                self.objp_list.append(np.asarray(o, np.float32))
                self.imgp_list.append(np.asarray(i, np.float32))

        self.epochs = epochs
        self.device = device
        self.val_ratio = val_ratio
        self.loss_history = []
        self.val_loss_history = []

    def __str__(self):
        return f"K:\n{self.mtx}\nDistortion: {self.model}"

    def _calibrate_K(self, imgp_list):
        """Calibrate K with zero distortion assumption."""
        flags = (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 |
                 cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
        _, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objp_list, imgp_list, self.image_size, None, None, flags=flags)

    def _reproject(self):
        """Reproject object points using K — these are the undistorted ground truth."""
        self.reprojected_imgp = []
        for objp, rvec, tvec in zip(self.objp_list, self.rvecs, self.tvecs):
            proj, _ = cv2.projectPoints(objp, rvec, tvec, self.mtx, self.dist)
            self.reprojected_imgp.append(proj.reshape(-1, 2))

    def _normalize(self, imgp_list):
        """Apply K_inv to pixel points → normalized coords."""
        cat = np.concatenate([p.reshape(-1, 1, 2) for p in imgp_list], axis=0).astype(np.float32)
        return cv2.undistortPoints(cat, self.mtx, self.dist).reshape(-1, 2)

    def _fit_distortion(self):
        """Train distortion model: distorted_normalized → reprojected_normalized.

        Uses 80/20 train/val split on points. Keeps best model by val loss.
        """
        # Input: distorted normalized (K_inv on original observed points)
        distorted_n = self._normalize(self.imgp_list)
        # Target: reprojected normalized (K_inv on reprojected points = ground truth)
        target_n = self._normalize(self.reprojected_imgp)

        # Train/val split
        n = len(distorted_n)
        perm = np.random.permutation(n)
        n_train = int(n * (1 - self.val_ratio))
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        x_train = torch.tensor(distorted_n[train_idx, 0], dtype=torch.float32, device=self.device)
        y_train = torch.tensor(distorted_n[train_idx, 1], dtype=torch.float32, device=self.device)
        tx_train = torch.tensor(target_n[train_idx, 0], dtype=torch.float32, device=self.device)
        ty_train = torch.tensor(target_n[train_idx, 1], dtype=torch.float32, device=self.device)

        x_val = torch.tensor(distorted_n[val_idx, 0], dtype=torch.float32, device=self.device)
        y_val = torch.tensor(distorted_n[val_idx, 1], dtype=torch.float32, device=self.device)
        tx_val = torch.tensor(target_n[val_idx, 0], dtype=torch.float32, device=self.device)
        ty_val = torch.tensor(target_n[val_idx, 1], dtype=torch.float32, device=self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        best_state = None

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            optimizer.zero_grad()
            px, py = self.model(x_train, y_train)
            loss = criterion(px, tx_train) + criterion(py, ty_train)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())

            # Validate
            self.model.eval()
            with torch.no_grad():
                vx, vy = self.model(x_val, y_val)
                val_loss = criterion(vx, tx_val) + criterion(vy, ty_val)
            self.val_loss_history.append(val_loss.item())

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if epoch % 2000 == 0:
                print(f"  epoch {epoch}: train={loss.item()} val={val_loss.item()}")

        # Restore best model
        self.model.load_state_dict(best_state)
        self.model.eval()
        print(f"  best val loss: {best_val_loss:.6f}")

    def undistort_points(self, imgp_list=None):
        """Undistort image points via forward pass: distorted_normalized → undistorted_normalized → pixels."""
        if imgp_list is None:
            imgp_list = self.imgp_list

        self.model.eval()
        fx, fy = self.mtx[0, 0], self.mtx[1, 1]
        cx, cy = self.mtx[0, 2], self.mtx[1, 2]

        lengths = [len(p.reshape(-1, 2)) for p in imgp_list]
        distorted = self._normalize(imgp_list)

        with torch.no_grad():
            x_t = torch.tensor(distorted[:, 0], dtype=torch.float32, device=self.device)
            y_t = torch.tensor(distorted[:, 1], dtype=torch.float32, device=self.device)
            x_u, y_u = self.model(x_t, y_t)

        u = x_u.cpu().numpy() * fx + cx
        v = y_u.cpu().numpy() * fy + cy
        all_undist = np.stack([u, v], axis=1).astype(np.float32)

        result = []
        offset = 0
        for n in lengths:
            result.append(all_undist[offset:offset + n])
            offset += n
        return result

    def reprojection_error(self):
        undist = self.undistort_points()
        sse = n = 0
        for ud, reproj in zip(undist, self.reprojected_imgp):
            d = reproj.astype(np.float64) - ud.reshape(-1, 2).astype(np.float64)
            sse += float((d * d).sum())
            n += d.size // 2
        return float(np.sqrt(sse / n))

    def fit(self):
        """Pipeline: K(no dist) → reproject → train distortion → undistort → recalibrate K."""
        # Step 1: Calibrate K assuming no distortion
        print("Step 1: Calibrate K (no distortion)")
        self._calibrate_K(self.imgp_list)
        self._reproject()

        # Step 2: Train distortion model (distorted → reprojected)
        print("Step 2: Train distortion model")
        self._fit_distortion()

        # Step 3: Undistort and recalibrate K
        print("Step 3: Recalibrate K with undistorted points")
        undistorted = self.undistort_points()
        self._calibrate_K(undistorted)
        self._reproject()

        print(f"Final RMSE: {self.reprojection_error()}")
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
        # print(f"Loss plot saved to {save_path}")

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
        # print(f"Distortion field saved to {save_path}")

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
        # print(f"Reprojection error distribution saved to {save_path}")
        # print(f"  Total points: {len(all_sq_dists)}, "
        #       f"Mean: {np.mean(all_sq_dists):.4f} px², "
        #       f"Median: {np.median(all_sq_dists):.4f} px², "
        #       f"95th pct: {p95:.4f} px², "
        #       f"Max: {np.max(all_sq_dists):.4f} px²")

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
                r2 = x**2 + y**2
                radial = 1 + self.k1 * r2 + self.k2 * r2**2 + self.k3 * r2**3
                x_out = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x**2)
                y_out = y * radial + self.p1 * (r2 + 2 * y**2) + 2 * self.p2 * x * y
                return x_out, y_out

            def __str__(self):
                return f"{self.__class__.__name__}(k1={self.k1.item():.4f}, k2={self.k2.item():.4f}, k3={self.k3.item():.4f}, p1={self.p1.item():.4f}, p2={self.p2.item():.4f})"

        self.model = OrdinaryPolynomialDistortion().to(self.device)

    def mlp_distortion(self):
        class MLPDistortion(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(2, 100),
                    nn.ReLU(),
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100, 25),
                    nn.ReLU(),
                    nn.Linear(25, 2),
                )
            def forward(self, x, y):
                out = self.mlp(torch.stack([x, y], dim=1))
                return out[:, 0], out[:, 1]
            def __str__(self):
                return f"{self.__class__.__name__}(mlp={self.mlp})"

        self.model = MLPDistortion().to(self.device)

    def save(self, path):
        """Save K matrix and distortion model state_dict."""
        torch.save({
            "mtx": self.mtx,
            "dist": self.dist,
            "image_size": self.image_size,
            "model_name": self.model.__class__.__name__,
            "model_state": self.model.state_dict(),
            "rmse": self.reprojection_error(),
        }, path)

    @classmethod
    def load(cls, path, device='cpu'):
        """Load a saved calibration result for inference (undistortion / PnP)."""
        data = torch.load(path, map_location=device, weights_only=False)
        obj = cls.__new__(cls)
        obj.mtx = data["mtx"]
        obj.dist = data["dist"]
        obj.image_size = data["image_size"]
        obj.device = device
        # Reconstruct model
        if data["model_name"] == "OrdinaryPolynomialDistortion":
            obj.ordinary_polynomial_distortion()
        elif data["model_name"] == "MLPDistortion":
            obj.mlp_distortion()
        obj.model.load_state_dict(data["model_state"])
        obj.model.eval()
        return obj


def calibrate_camera(points_file, output_dir):
    """Run all calibration methods on one camera's point file and save results."""
    from pathlib import Path
    data = np.load(points_file, allow_pickle=True)
    obj_points = [arr.astype(np.float32) for arr in data["obj_points"]]
    img_points = [arr.astype(np.float32) for arr in data["img_points"]]
    img_size = tuple(int(x) for x in data["img_size"])
    cam_name = Path(points_file).stem

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # OpenCV full calibration
    opencv_result = opencv_full_calib(obj_points, img_points, img_size)
    np.savez(str(out / f"{cam_name}_opencv.npz"),
             K=opencv_result["K"], dist=opencv_result["dist"],
             img_size=np.array(img_size), rmse=opencv_result["rmse_px"])

    # Ordinary polynomial distortion
    poly_fit = pytorch_distortion_fit(obj_points, img_points, img_size)
    poly_fit.ordinary_polynomial_distortion()
    poly_fit.fit()
    poly_fit.save(str(out / f"{cam_name}_poly.pt"))

    # MLP distortion
    mlp_fit = pytorch_distortion_fit(obj_points, img_points, img_size)
    mlp_fit.mlp_distortion()
    mlp_fit.fit()
    mlp_fit.save(str(out / f"{cam_name}_mlp.pt"))

    print(f"\n{cam_name} results:")
    print(f"  OpenCV RMSE: {opencv_result['rmse_px']:.6f}")
    print(f"  Poly   RMSE: {poly_fit.reprojection_error():.6f}")
    print(f"  MLP    RMSE: {mlp_fit.reprojection_error():.6f}")

    return opencv_result, poly_fit, mlp_fit


if __name__ == "__main__":
    from pathlib import Path
    import sys

    points_dir = Path("../points/water")
    output_dir = Path("../calibration/water")

    for npz in sorted(points_dir.glob("cam*.npz")):
        calibrate_camera(str(npz), str(output_dir))