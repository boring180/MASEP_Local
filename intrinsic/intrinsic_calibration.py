"""Pinhole OpenCV calibration + PyTorch distortion fit; full OpenCV calib as reference."""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

def opencv_full_calib(objpoints, imgpoints, image_size):
    # Minimal filter: OpenCV requires >=4 non-collinear points per view.
    objp_list, imgp_list = [], []
    for o, i in zip(objpoints, imgpoints):
        if len(i) < 4:
            continue
        xy = np.asarray(o, np.float64)[:, :2]
        s = np.linalg.svd(xy - xy.mean(0), compute_uv=False)
        if s[1] < 1e-6 * s[0]:
            continue
        objp_list.append(np.asarray(o, np.float32))
        imgp_list.append(np.asarray(i, np.float32))

    rms_opencv, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objp_list, imgp_list, image_size, None, None)

    return {
        "K": mtx,
        "dist": dist.reshape(-1).astype(np.float64),
        "rvecs": rvecs,
        "tvecs": tvecs,
        "rmse_px": float(rms_opencv),
        "filtered_obj": objp_list,
        "filtered_img": imgp_list,
        "rms_opencv": float(rms_opencv),
    }


class pytorch_distortion_fit():
    def __init__(self, objpoints, imgpoints, image_size, K, dist, rvecs, tvecs,
                 device='cpu', epochs=100000, val_ratio=0.2):
        """Fit a nonlinear distortion model on top of a pre-calibrated K + D.

        K and the OpenCV-calibrated distortion D are taken as-is and used for
        both per-view reprojection (target) and pixel normalization (input).
        The nonlinear model therefore learns only the RESIDUAL distortion left
        after the OpenCV lens model has been applied — for air data this
        residual is near zero; for water data this captures the cross-medium
        refractive effect on top of the air calibration.
        If rvecs/tvecs are omitted they are recovered per view via solvePnP.
        """
        self.image_size = image_size
        self.mtx = np.asarray(K, dtype=np.float64)
        # Use the OpenCV-calibrated distortion throughout — reprojections and
        # normalized-coord transforms both go through this D.
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1)

        # Caller must pass obj/img lists aligned 1:1 with rvecs/tvecs.
        self.objp_list = [np.asarray(o, np.float32) for o in objpoints]
        self.imgp_list = [np.asarray(i, np.float32) for i in imgpoints]

        self.rvecs = list(rvecs)
        self.tvecs = list(tvecs)

        self.epochs = epochs
        self.device = device
        self.val_ratio = val_ratio
        self.loss_history = []
        self.val_loss_history = []

    def __str__(self):
        return f"K:\n{self.mtx}\nDistortion: {self.model}"

    def _reproject(self):
        """Reproject object points through K and the OpenCV-calibrated D — target image points."""
        self.reprojected_imgp = []
        for objp, rvec, tvec in zip(self.objp_list, self.rvecs, self.tvecs):
            proj, _ = cv2.projectPoints(objp, rvec, tvec, self.mtx, self.dist)
            self.reprojected_imgp.append(proj.reshape(-1, 2))

    def _normalize(self, imgp_list):
        """Apply K_inv + OpenCV lens-undistort to pixel points → normalized coords."""
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
        """Pipeline: use provided K (+linear D) → train nonlinear distortion model."""
        # K is fixed; only build the undistorted-target reprojection and train.
        print("Step 1: Reproject with provided K (zero distortion target)")
        self._reproject()

        print("Step 2: Train distortion model")
        self._fit_distortion()

        print(f"Final RMSE: {self.reprojection_error()}")
        return self.mtx, self.dist, self.rvecs, self.tvecs

    def plot_loss(self, save_path="loss_plot.png", opencv_rmse=None):
        import matplotlib.pyplot as plt
        name = self.model.__class__.__name__
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for ax in (ax1, ax2):
            ax.plot(self.loss_history, label="train", color="steelblue")
            ax.plot(self.val_loss_history, label="val", color="darkorange")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True)
            if opencv_rmse is not None:
                ax.axhline(y=opencv_rmse, color="green", linestyle="-", linewidth=2,
                           label=f"OpenCV RMSE ({opencv_rmse:.4f})")
            ax.legend()

        ax1.set_title(f"{name} Training/Val Loss")
        ax2.set_yscale("log")
        ax2.set_title(f"{name} Training/Val Loss (log)")

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

        undist_list = self.undistort_points()
        all_sq_dists = []
        for imgp, reproj in zip(undist_list, self.reprojected_imgp):
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
        ax.set_title(f"PDF of Per-Point Squared Reprojection Error — {self.model.__class__.__name__} "
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
                    nn.Tanh(),
                    nn.Linear(100, 100),
                    nn.Tanh(),
                    nn.Linear(100, 25),
                    nn.Tanh(),
                    nn.Linear(25, 2),
                )
            def forward(self, x, y):
                out = self.mlp(torch.stack([x, y], dim=1))
                return out[:, 0], out[:, 1]
            def __str__(self):
                return f"{self.__class__.__name__}(mlp={self.mlp})"

        self.model = MLPDistortion().to(self.device)

    def refractive_distortion(self, n_init=1.33):
        """Physics-based flat-port refractive model (Singh & Alexis, 2024).

        Single learnable refractive index n. Backward (inverse) mapping —
        same direction as the poly/MLP models: input (x, y) is the distorted
        normalized point p̄_C,r (after K_inv + OpenCV lens-undistort on the
        observed water pixel), output is the undistorted air-equivalent p̄_C.
        Implements the paper's Eq. 9:
            p̄_C = p̄_C,r / sqrt(n² r_r² + n² - r_r²),  r_r = ‖p̄_C,r‖.
        After fitting, the learned n should be close to the medium's true
        refractive index (≈1.33 for water at 20 °C).
        """
        class RefractiveDistortion(nn.Module):
            def __init__(self, n0):
                super().__init__()
                self.n = nn.Parameter(torch.tensor(float(n0)))

            def forward(self, x, y):
                # x, y = distorted normalized coords (p̄_C,r); return undistorted p̄_C.
                r2 = x * x + y * y
                m = torch.sqrt(self.n ** 2 * r2 + self.n ** 2 - r2)
                return x / m, y / m

            def __str__(self):
                return f"RefractiveDistortion(n={self.n.item():.6f})"

        self.model = RefractiveDistortion(n_init).to(self.device)

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
        elif data["model_name"] == "RefractiveDistortion":
            obj.refractive_distortion()
        obj.model.load_state_dict(data["model_state"])
        obj.model.eval()
        return obj


def opencv_calibrate_and_save(points_file, output_dir):
    """Run full OpenCV calibration on `points_file` and persist K/D/poses.

    Writes `{cam_name}_opencv.npz` under `output_dir` containing K, dist,
    rvecs, tvecs, img_size, and rmse. Returns (opencv_result, save_path).
    """
    from pathlib import Path
    data = np.load(points_file, allow_pickle=True)
    obj_points = [arr.astype(np.float32) for arr in data["obj_points"]]
    img_points = [arr.astype(np.float32) for arr in data["img_points"]]
    img_size = tuple(int(x) for x in data["img_size"])
    cam_name = Path(points_file).stem

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = opencv_full_calib(obj_points, img_points, img_size)
    save_path = out / f"{cam_name}_opencv.npz"
    np.savez(str(save_path),
             K=result["K"], dist=result["dist"],
             rvecs=np.array(result["rvecs"], dtype=object),
             tvecs=np.array(result["tvecs"], dtype=object),
             filtered_obj=np.array(result["filtered_obj"], dtype=object),
             filtered_img=np.array(result["filtered_img"], dtype=object),
             img_size=np.array(img_size), rmse=result["rmse_px"])
    print(f"OpenCV calibration saved to {save_path}")
    print(f"  K =\n{result['K']}")
    print(f"  dist = {result['dist']}")
    print(f"  RMSE = {result['rmse_px']:.6f} px")
    return result, save_path


def load_opencv_result(path):
    """Load a saved OpenCV calibration npz back into the dict shape of opencv_full_calib."""
    data = np.load(path, allow_pickle=True)
    out = {
        "K": np.asarray(data["K"], dtype=np.float64),
        "dist": np.asarray(data["dist"], dtype=np.float64).reshape(-1),
        "rvecs": [np.asarray(r, dtype=np.float64) for r in data["rvecs"]],
        "tvecs": [np.asarray(t, dtype=np.float64) for t in data["tvecs"]],
        "rmse_px": float(data["rmse"]),
    }
    if "filtered_obj" in data.files:
        out["filtered_obj"] = [np.asarray(a, np.float32) for a in data["filtered_obj"]]
        out["filtered_img"] = [np.asarray(a, np.float32) for a in data["filtered_img"]]
    return out


def save_reprojection_images(points_file, opencv_result, output_dir):
    """Render per-view reprojection overlays (observed vs projected points).

    For each view in `points_file`, projects the object points through the
    calibrated K/D and saved pose, then draws both observed (green) and
    reprojected (red) points on a blank canvas of `img_size`. Writes one
    PNG per view under `output_dir/{cam_name}/view_XXX.png`.
    """
    from pathlib import Path
    data = np.load(points_file, allow_pickle=True)
    obj_points = [arr.astype(np.float32) for arr in data["obj_points"]]
    img_points = [arr.astype(np.float32) for arr in data["img_points"]]
    img_size = tuple(int(x) for x in data["img_size"])
    cam_name = Path(points_file).stem
    w, h = img_size

    K = opencv_result["K"]
    dist = opencv_result["dist"]
    rvecs = opencv_result["rvecs"]
    tvecs = opencv_result["tvecs"]

    # OpenCV calibration filters out weak views, so rvecs/tvecs may be fewer
    # than the raw input. Re-solve PnP per-view to pair each view with a pose.
    out = Path(output_dir) / cam_name
    out.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for idx, (objp, imgp) in enumerate(zip(obj_points, img_points)):
        imgp_arr = np.asarray(imgp, dtype=np.float32).reshape(-1, 2)
        if len(imgp_arr) < 4:
            continue
        ok, rvec, tvec = cv2.solvePnP(
            objp.reshape(-1, 1, 3), imgp_arr.reshape(-1, 1, 2), K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        if not np.all(np.isfinite(proj)):
            # Unstable pose (this view was likely dropped as an outlier during calibration).
            continue

        def pt(xy):
            return (int(round(float(xy[0]))), int(round(float(xy[1]))))

        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        for row in imgp_arr:
            cv2.circle(canvas, pt(row), 4, (0, 200, 0), 2)
        for row in proj:
            cv2.circle(canvas, pt(row), 3, (0, 0, 255), -1)
        for obs, pr in zip(imgp_arr, proj):
            cv2.line(canvas, pt(obs), pt(pr), (128, 128, 128), 1)

        d = proj.astype(np.float64) - imgp_arr.astype(np.float64)
        rmse = float(np.sqrt((d * d).sum(1).mean()))
        cv2.putText(canvas, f"view {idx}  RMSE={rmse:.3f}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, "green=observed  red=reprojected",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        cv2.imwrite(str(out / f"view_{idx:03d}.png"), canvas)
        n_saved += 1

    print(f"Saved {n_saved} reprojection images to {out}")


def calibrate_camera_multiple_methods(points_file, output_dir, opencv_dir=None):
    """Fit poly + MLP distortion on one camera's point file.

    If `opencv_dir` is None, run full OpenCV calibration (saved into
    `output_dir`) and use its K/D/poses. Otherwise load the existing
    `{cam_name}_opencv.npz` from `opencv_dir` and reuse it.
    """
    from pathlib import Path
    data = np.load(points_file, allow_pickle=True)
    obj_points = [arr.astype(np.float32) for arr in data["obj_points"]]
    img_points = [arr.astype(np.float32) for arr in data["img_points"]]
    img_size = tuple(int(x) for x in data["img_size"])
    cam_name = Path(points_file).stem

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if opencv_dir is None:
        opencv_result, _ = opencv_calibrate_and_save(points_file, output_dir)
    else:
        opencv_result = load_opencv_result(Path(opencv_dir) / f"{cam_name}_opencv.npz")

    opencv_rmse = opencv_result["rmse_px"]
    fit_obj = opencv_result["filtered_obj"]
    fit_img = opencv_result["filtered_img"]

    # Ordinary polynomial distortion (K + D from OpenCV)
    poly_fit = pytorch_distortion_fit(
        fit_obj, fit_img, img_size,
        K=opencv_result["K"], dist=opencv_result["dist"],
        rvecs=opencv_result["rvecs"], tvecs=opencv_result["tvecs"])
    poly_fit.ordinary_polynomial_distortion()
    poly_fit.fit()
    poly_fit.save(str(out / f"{cam_name}_poly.pt"))
    poly_fit.plot_loss(str(out / f"{cam_name}_poly_loss.png"), opencv_rmse=opencv_rmse)
    poly_fit.plot_distortion_field(str(out / f"{cam_name}_poly_distortion_field.png"))
    poly_fit.plot_reprojection_error_distribution(str(out / f"{cam_name}_poly_reproj_err.png"))

    # MLP distortion (K + D from OpenCV)
    mlp_fit = pytorch_distortion_fit(
        fit_obj, fit_img, img_size,
        K=opencv_result["K"], dist=opencv_result["dist"],
        rvecs=opencv_result["rvecs"], tvecs=opencv_result["tvecs"])
    mlp_fit.mlp_distortion()
    mlp_fit.fit()
    mlp_fit.save(str(out / f"{cam_name}_mlp.pt"))
    mlp_fit.plot_loss(str(out / f"{cam_name}_mlp_loss.png"), opencv_rmse=opencv_rmse)
    mlp_fit.plot_distortion_field(str(out / f"{cam_name}_mlp_distortion_field.png"))
    mlp_fit.plot_reprojection_error_distribution(str(out / f"{cam_name}_mlp_reproj_err.png"))

    print(f"\n{cam_name} results:")
    print(f"  OpenCV RMSE: {opencv_result['rmse_px']:.6f}")
    print(f"  Poly   RMSE: {poly_fit.reprojection_error():.6f}")
    print(f"  MLP    RMSE: {mlp_fit.reprojection_error():.6f}")

    return opencv_result, poly_fit, mlp_fit


def _direct_reprojection_rmse(obj_points, img_points, K, dist):
    """Per-view PnP with given K/dist, then measure RMSE against reprojection.

    Returns (rmse_px, rvecs, tvecs, filt_obj, filt_img) with all four lists
    aligned 1:1 to the views that passed PnP.
    """
    rvecs, tvecs, filt_obj, filt_img = [], [], [], []
    sse = n = 0
    for objp, imgp in zip(obj_points, img_points):
        imgp_arr = np.asarray(imgp, np.float32).reshape(-1, 2)
        objp_arr = np.asarray(objp, np.float32).reshape(-1, 3)
        if len(imgp_arr) < 4:
            continue
        ok, rvec, tvec = cv2.solvePnP(
            objp_arr.reshape(-1, 1, 3), imgp_arr.reshape(-1, 1, 2), K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(objp_arr, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        if not np.all(np.isfinite(proj)):
            continue
        d = proj.astype(np.float64) - imgp_arr.astype(np.float64)
        sse += float((d * d).sum())
        n += d.shape[0]
        rvecs.append(rvec)
        tvecs.append(tvec)
        filt_obj.append(objp_arr)
        filt_img.append(imgp_arr)
    rmse = float(np.sqrt(sse / n)) if n else float("nan")
    return rmse, rvecs, tvecs, filt_obj, filt_img


def calibrate_water_from_air(water_points_file, air_opencv_path, output_dir):
    """Compare 5 methods on water points given an air calibration:
      1. Direct use of air intrinsics (PnP + reproject with air K/D).
      2. Full OpenCV recalibration on water points.
      3. Polynomial cross-medium correction on top of air K/D.
      4. MLP cross-medium correction on top of air K/D.
      5. Physics-based refractive-index fit (Singh & Alexis, 2024).

    Methods 3–5 reuse the per-view PnP poses from Method 1 so every fit
    operates on the same view set, letting the learned residual isolate the
    cross-medium distortion instead of pose error.
    """
    from pathlib import Path
    data = np.load(water_points_file, allow_pickle=True)
    obj_points = [arr.astype(np.float32) for arr in data["obj_points"]]
    img_points = [arr.astype(np.float32) for arr in data["img_points"]]
    img_size = tuple(int(x) for x in data["img_size"])
    cam_name = Path(water_points_file).stem

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    air = load_opencv_result(air_opencv_path)
    air_K, air_dist = air["K"], air["dist"]

    # Method 1: direct air intrinsics on water points. Provides shared
    # rvecs/tvecs (and the matching obj/img list) used by methods 3–5.
    m1_rmse, m1_rvecs, m1_tvecs, fit_obj, fit_img = _direct_reprojection_rmse(
        obj_points, img_points, air_K, air_dist)
    np.savez(str(out / f"{cam_name}_air_direct.npz"),
             K=air_K, dist=air_dist,
             rvecs=np.array(m1_rvecs, dtype=object),
             tvecs=np.array(m1_tvecs, dtype=object),
             img_size=np.array(img_size), rmse=m1_rmse)

    # Method 2: full OpenCV recalibration on water points.
    m2 = opencv_full_calib(obj_points, img_points, img_size)
    np.savez(str(out / f"{cam_name}_water_opencv.npz"),
             K=m2["K"], dist=m2["dist"],
             rvecs=np.array(m2["rvecs"], dtype=object),
             tvecs=np.array(m2["tvecs"], dtype=object),
             filtered_obj=np.array(m2["filtered_obj"], dtype=object),
             filtered_img=np.array(m2["filtered_img"], dtype=object),
             img_size=np.array(img_size), rmse=m2["rmse_px"])

    def _new_fit():
        return pytorch_distortion_fit(
            fit_obj, fit_img, img_size,
            K=air_K, dist=air_dist,
            rvecs=m1_rvecs, tvecs=m1_tvecs)

    # Method 3: polynomial residual correction.
    poly_fit = _new_fit()
    poly_fit.ordinary_polynomial_distortion()
    poly_fit.fit()
    poly_fit.save(str(out / f"{cam_name}_poly.pt"))
    poly_fit.plot_loss(str(out / f"{cam_name}_poly_loss.png"), opencv_rmse=m1_rmse)
    poly_fit.plot_distortion_field(str(out / f"{cam_name}_poly_distortion_field.png"))
    poly_fit.plot_reprojection_error_distribution(str(out / f"{cam_name}_poly_reproj_err.png"))

    # Method 4: MLP residual correction.
    mlp_fit = _new_fit()
    mlp_fit.mlp_distortion()
    mlp_fit.fit()
    mlp_fit.save(str(out / f"{cam_name}_mlp.pt"))
    mlp_fit.plot_loss(str(out / f"{cam_name}_mlp_loss.png"), opencv_rmse=m1_rmse)
    mlp_fit.plot_distortion_field(str(out / f"{cam_name}_mlp_distortion_field.png"))
    mlp_fit.plot_reprojection_error_distribution(str(out / f"{cam_name}_mlp_reproj_err.png"))

    # Method 5: physics-based refractive index fit.
    refr_fit = _new_fit()
    refr_fit.refractive_distortion()
    refr_fit.fit()
    refr_fit.save(str(out / f"{cam_name}_refractive.pt"))
    refr_fit.plot_loss(str(out / f"{cam_name}_refractive_loss.png"), opencv_rmse=m1_rmse)
    refr_fit.plot_distortion_field(str(out / f"{cam_name}_refractive_distortion_field.png"))
    refr_fit.plot_reprojection_error_distribution(str(out / f"{cam_name}_refractive_reproj_err.png"))
    learned_n = float(refr_fit.model.n.detach().cpu().item())

    summary = {
        "air_direct": m1_rmse,
        "water_opencv": m2["rmse_px"],
        "poly_correction": poly_fit.reprojection_error(),
        "mlp_correction": mlp_fit.reprojection_error(),
        "refractive": refr_fit.reprojection_error(),
    }
    print(f"\n{cam_name} cross-medium comparison:")
    for k, v in summary.items():
        print(f"  {k:20s} RMSE = {v:.6f} px")
    print(f"  learned refractive index n = {learned_n:.6f} (water ≈ 1.33)")
    summary["learned_n"] = learned_n
    return summary


if __name__ == "__main__":
    from pathlib import Path

    air_points_dir = Path("../points/air")
    air_output_dir = Path("../calibration/air")
    water_points_dir = Path("../points/water")
    water_output_dir = Path("../calibration/water")
    reproj_dir = Path("../points/air_reprojected")

    # for npz in sorted(air_points_dir.glob("cam*.npz")):
    #     calibrate_camera_multiple_methods(str(npz), str(air_output_dir))
        # save_reprojection_images(str(npz), opencv_result, str(reproj_dir))

    for npz in sorted(water_points_dir.glob("cam*.npz")):
        cam_name = npz.stem
        air_opencv_path = air_output_dir / f"{cam_name}_opencv.npz"
        if not air_opencv_path.exists():
            print(f"Skipping {cam_name}: no air calibration at {air_opencv_path}")
            continue
        calibrate_water_from_air(str(npz), str(air_opencv_path), str(water_output_dir))