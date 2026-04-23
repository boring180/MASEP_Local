"""Legacy — PyTorch polynomial / MLP distortion fit on top of an OpenCV K/D.

Superseded by the refractive bundle adjustment in `intrinsic_calibration.py`.
Kept for reference; not imported by the active pipeline.

Input/target pairs are both in normalized coords after K_inv + OpenCV
lens-undistort, so the learned model captures only the residual distortion.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim


# Pick the best available device. MPS requires float32.
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE = torch.float32
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float64


# --- distortion model factories ------------------------------------------------

class _Polynomial(nn.Module):
    """Brown–Conrady polynomial (backward mapping: distorted → undistorted)."""

    def __init__(self):
        super().__init__()
        self.k1 = nn.Parameter(torch.tensor(0.0))
        self.k2 = nn.Parameter(torch.tensor(0.0))
        self.k3 = nn.Parameter(torch.tensor(0.0))
        self.p1 = nn.Parameter(torch.tensor(0.0))
        self.p2 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, y):
        r2 = x * x + y * y
        radial = 1 + self.k1 * r2 + self.k2 * r2 ** 2 + self.k3 * r2 ** 3
        xo = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        yo = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        return xo, yo


class _MLP(nn.Module):
    """Small MLP (backward mapping: distorted → undistorted)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 25), nn.Tanh(),
            nn.Linear(25, 2),
        )

    def forward(self, x, y):
        out = self.net(torch.stack([x, y], dim=1))
        return out[:, 0], out[:, 1]


def make_model(kind):
    if kind == "poly":
        return _Polynomial()
    if kind == "mlp":
        return _MLP()
    raise ValueError(f"unknown model kind: {kind!r}")


# --- main class ----------------------------------------------------------------

class PytorchDistortionFit:
    """Fit a residual distortion model on top of an OpenCV (K, D, poses)."""

    def __init__(self, obj_points, img_points, image_size, K, dist, rvecs, tvecs,
                 kind="poly", epochs=10000, val_ratio=0.2, lr=0.01,
                 jacobian_lambda=1.0):
        self.image_size = image_size
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        self.obj_list = [np.asarray(o, np.float32) for o in obj_points]
        self.img_list = [np.asarray(i, np.float32) for i in img_points]
        self.rvecs = list(rvecs)
        self.tvecs = list(tvecs)

        self.epochs = epochs
        self.val_ratio = val_ratio
        self.lr = lr
        self.jacobian_lambda = jacobian_lambda
        self.kind = kind
        self.model = make_model(kind).to(DEVICE, dtype=DTYPE)

        self.train_rmse = []
        self.val_rmse = []
        self.reprojected_imgp = None

    # ---- OpenCV-side geometry ------------------------------------------------

    def _reproject(self):
        """Project obj_points through (K, D) with the stored poses."""
        self.reprojected_imgp = [
            cv2.projectPoints(o, r, t, self.K, self.dist)[0].reshape(-1, 2)
            for o, r, t in zip(self.obj_list, self.rvecs, self.tvecs)
        ]

    def _normalize(self, imgp_list):
        """Pixels → K_inv + lens-undistorted normalized coords."""
        cat = np.concatenate([p.reshape(-1, 1, 2) for p in imgp_list], axis=0)
        return cv2.undistortPoints(cat.astype(np.float32), self.K, self.dist).reshape(-1, 2)

    # ---- training ------------------------------------------------------------

    def _to_tensor(self, arr):
        return torch.as_tensor(arr, dtype=DTYPE, device=DEVICE)

    def _jacobian_penalty(self, x, y):
        """Punish det(J) <= 0 so the learned map stays locally one-to-one."""
        x = x.detach().clone().requires_grad_(True)
        y = y.detach().clone().requires_grad_(True)
        fx, fy = self.model(x, y)
        (dxdx, dxdy) = torch.autograd.grad(fx.sum(), [x, y], create_graph=True)
        (dydx, dydy) = torch.autograd.grad(fy.sum(), [x, y], create_graph=True)
        det = dxdx * dydy - dxdy * dydx
        return torch.relu(1e-3 - det).pow(2).mean()

    def _fit(self):
        if self.reprojected_imgp is None:
            self._reproject()

        src = self._normalize(self.img_list)            # distorted normalized
        tgt = self._normalize(self.reprojected_imgp)    # target normalized

        n = len(src)
        perm = np.random.permutation(n)
        n_tr = int(n * (1 - self.val_ratio))
        tr, vl = perm[:n_tr], perm[n_tr:]

        xs, ys = self._to_tensor(src[:, 0]), self._to_tensor(src[:, 1])
        xt, yt = self._to_tensor(tgt[:, 0]), self._to_tensor(tgt[:, 1])
        tr_i = torch.as_tensor(tr, device=DEVICE)
        vl_i = torch.as_tensor(vl, device=DEVICE)

        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        mse = nn.MSELoss()
        best_val = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            opt.zero_grad()
            px, py = self.model(xs[tr_i], ys[tr_i])
            fit_loss = mse(px, xt[tr_i]) + mse(py, yt[tr_i])
            loss = fit_loss + (self.jacobian_lambda * self._jacobian_penalty(xs[tr_i], ys[tr_i])
                               if self.jacobian_lambda > 0 else 0.0)
            loss.backward()
            opt.step()

            self.model.eval()
            with torch.no_grad():
                vx, vy = self.model(xs[vl_i], ys[vl_i])
                v_loss = mse(vx, xt[vl_i]) + mse(vy, yt[vl_i])

            self.train_rmse.append(float(fit_loss.item()) ** 0.5)
            self.val_rmse.append(float(v_loss.item()) ** 0.5)

            if v_loss.item() < best_val:
                best_val = float(v_loss.item())
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if epoch % 2000 == 0:
                print(f"  epoch {epoch}: train_rmse={self.train_rmse[-1]:.4f} "
                      f"val_rmse={self.val_rmse[-1]:.4f}")

        self.model.load_state_dict(best_state)
        self.model.eval()
        print(f"  best val rmse: {best_val ** 0.5:.6f}")

    def fit(self):
        self._reproject()
        self._fit()
        print(f"Final RMSE: {self.reprojection_error():.6f}")
        return self

    # ---- inference -----------------------------------------------------------

    def undistort_points(self, imgp_list=None):
        """Apply the learned model to recover undistorted pixels."""
        imgp_list = imgp_list if imgp_list is not None else self.img_list
        lens = [len(p.reshape(-1, 2)) for p in imgp_list]
        src = self._normalize(imgp_list)
        with torch.no_grad():
            x = self._to_tensor(src[:, 0])
            y = self._to_tensor(src[:, 1])
            xu, yu = self.model(x, y)
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        u = xu.cpu().numpy() * fx + cx
        v = yu.cpu().numpy() * fy + cy
        all_u = np.stack([u, v], axis=1).astype(np.float32)
        out, off = [], 0
        for n in lens:
            out.append(all_u[off:off + n])
            off += n
        return out

    def reprojection_error(self):
        undist = self.undistort_points()
        sse = n = 0
        for u, r in zip(undist, self.reprojected_imgp):
            d = r.astype(np.float64) - u.astype(np.float64)
            sse += float((d * d).sum())
            n += d.shape[0]
        return float(np.sqrt(sse / n))

    # ---- IO ------------------------------------------------------------------

    def save(self, path):
        torch.save({
            "K": self.K, "dist": self.dist, "image_size": self.image_size,
            "kind": self.kind, "state": self.model.state_dict(),
            "rmse": self.reprojection_error(),
        }, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location=DEVICE, weights_only=False)
        obj = cls.__new__(cls)
        obj.K = data["K"]; obj.dist = data["dist"]
        obj.image_size = data["image_size"]
        obj.kind = data["kind"]
        obj.model = make_model(obj.kind).to(DEVICE, dtype=DTYPE)
        obj.model.load_state_dict(data["state"])
        obj.model.eval()
        return obj

    # ---- plots ---------------------------------------------------------------

    def plot_loss(self, save_path, opencv_rmse=None):
        import matplotlib.pyplot as plt
        name = self.model.__class__.__name__
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
        for ax in (a1, a2):
            ax.plot(self.train_rmse, label="train", color="steelblue")
            ax.plot(self.val_rmse, label="val", color="darkorange")
            if opencv_rmse is not None:
                ax.axhline(opencv_rmse, color="green", linewidth=2,
                           label=f"OpenCV RMSE ({opencv_rmse:.4f})")
            ax.set_xlabel("epoch"); ax.set_ylabel("RMSE (normalized)")
            ax.grid(True); ax.legend()
        a1.set_title(f"{name} train/val RMSE")
        a2.set_yscale("log"); a2.set_title(f"{name} RMSE (log)")
        plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close(fig)

    def plot_distortion_field(self, save_path, grid_n=20):
        import matplotlib.pyplot as plt
        w, h = self.image_size
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        uu, vv = np.meshgrid(np.linspace(0, w, grid_n), np.linspace(0, h, grid_n))
        uf, vf = uu.flatten(), vv.flatten()
        with torch.no_grad():
            xu, yu = self.model(self._to_tensor((uf - cx) / fx),
                                self._to_tensor((vf - cy) / fy))
        du = xu.cpu().numpy() * fx + cx - uf
        dv = yu.cpu().numpy() * fy + cy - vf
        scale = 50.0
        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(len(uf)):
            ax.annotate("", xy=(uf[i] + du[i] * scale, vf[i] + dv[i] * scale),
                        xytext=(uf[i], vf[i]),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_aspect("equal")
        ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
        ax.set_title(f"Distortion field ({self.model.__class__.__name__}, arrows ×{scale:.0f})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close(fig)

    def plot_reprojection_error_distribution(self, save_path):
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
        undist = self.undistort_points()
        sq = np.concatenate([
            ((r.astype(np.float64) - u.astype(np.float64)) ** 2).sum(axis=1)
            for u, r in zip(undist, self.reprojected_imgp)
        ])
        p95 = np.percentile(sq, 95)
        clip = sq[sq <= p95]
        kde = gaussian_kde(clip)
        x = np.linspace(0, p95, 500)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(x, kde(x), alpha=0.3, color="steelblue")
        ax.plot(x, kde(x), color="steelblue", linewidth=2)
        ax.axvline(sq.mean(), color="red", linestyle="--",
                   label=f"mean = {sq.mean():.4f} px²")
        ax.axvline(np.median(sq), color="orange", linestyle="--",
                   label=f"median = {np.median(sq):.4f} px²")
        ax.set_xlim(0, p95 * 1.05)
        ax.set_xlabel("squared reprojection distance (px²)")
        ax.set_ylabel("density")
        ax.set_title(f"Per-point squared error — {self.model.__class__.__name__} "
                     f"(95th pct, {len(sq) - len(clip)} outliers clipped)")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close(fig)
