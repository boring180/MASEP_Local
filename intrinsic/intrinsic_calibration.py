"""OpenCV pinhole calibration + cross-medium refractive bundle adjustment.

Standalone module: OpenCV calibration, on-disk I/O, and a PyTorch bundle
adjustment for the Singh & Alexis 2024 refractive camera model.
"""

from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim


# Pick the best available device. CUDA preferred;
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float64
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float64


# ---- OpenCV calibration -------------------------------------------------------

def opencv_full_calib(obj_points, img_points, image_size):
    """Run `cv2.calibrateCamera` after dropping degenerate views."""
    filt_obj, filt_img, kept = [], [], []
    for idx, (o, i) in enumerate(zip(obj_points, img_points)):
        if len(i) < 4:
            continue
        xy = np.asarray(o, np.float64)[:, :2]
        s = np.linalg.svd(xy - xy.mean(0), compute_uv=False)
        if s[1] < 1e-6 * s[0]:  # colinear
            continue
        filt_obj.append(np.asarray(o, np.float32))
        filt_img.append(np.asarray(i, np.float32))
        kept.append(idx)

    rmse, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        filt_obj, filt_img, image_size, None, None)
    return {
        "K": K,
        "dist": dist.reshape(-1).astype(np.float64),
        "rvecs": rvecs, "tvecs": tvecs,
        "filtered_obj": filt_obj, "filtered_img": filt_img,
        "kept_indices": kept,
        "rmse_px": float(rmse),
    }


def _split_points(obj_lists, img_lists, val_ratio=0.2, seed=0):
    """Per-view 80/20 point split. Returns (tr_obj, tr_img, vl_obj, vl_img)."""
    rng = np.random.default_rng(seed)
    tr_o, tr_i, vl_o, vl_i = [], [], [], []
    for obj, img in zip(obj_lists, img_lists):
        obj = np.asarray(obj, np.float32).reshape(-1, 3)
        img = np.asarray(img, np.float32).reshape(-1, 2)
        perm = rng.permutation(len(img))
        n_tr = max(1, int(len(img) * (1 - val_ratio)))
        tr_o.append(obj[perm[:n_tr]]); tr_i.append(img[perm[:n_tr]])
        vl_o.append(obj[perm[n_tr:]]); vl_i.append(img[perm[n_tr:]])
    return tr_o, tr_i, vl_o, vl_i


def _reprojection_rmse(obj_lists, img_lists, K, dist, rvecs, tvecs):
    """RMSE of cv2.projectPoints vs. observed pixels across aligned views."""
    sse = n = 0
    for obj, img, r, t in zip(obj_lists, img_lists, rvecs, tvecs):
        if len(img) == 0:
            continue
        proj = cv2.projectPoints(obj.reshape(-1, 1, 3), r, t, K, dist)[0].reshape(-1, 2)
        d = proj.astype(np.float64) - img.astype(np.float64)
        sse += float((d * d).sum()); n += d.shape[0]
    return float(np.sqrt(sse / n)) if n else float("nan")


def opencv_calibrate_and_save(points_file, output_dir):
    """Calibrate and persist K, D, poses, and the filtered point lists."""
    data = np.load(points_file, allow_pickle=True)
    obj_points = [a.astype(np.float32) for a in data["obj_points"]]
    img_points = [a.astype(np.float32) for a in data["img_points"]]
    img_size = tuple(int(x) for x in data["img_size"])
    cam = Path(points_file).stem

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    res = opencv_full_calib(obj_points, img_points, img_size)
    path = out / f"{cam}_opencv.npz"
    np.savez(str(path),
             K=res["K"], dist=res["dist"],
             rvecs=np.array(res["rvecs"], dtype=object),
             tvecs=np.array(res["tvecs"], dtype=object),
             filtered_obj=np.array(res["filtered_obj"], dtype=object),
             filtered_img=np.array(res["filtered_img"], dtype=object),
             img_size=np.array(img_size), rmse=res["rmse_px"])
    print(f"OpenCV calibration saved to {path}")
    print(f"  K =\n{res['K']}")
    print(f"  dist = {res['dist']}")
    print(f"  RMSE = {res['rmse_px']:.6f} px")
    return res, path


def load_opencv_result(path):
    """Inverse of `opencv_calibrate_and_save`."""
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


def _direct_reprojection_rmse(obj_points, img_points, K, dist):
    """Per-view PnP with given K/D; return RMSE + aligned obj/img/pose lists."""
    rvecs, tvecs, filt_obj, filt_img = [], [], [], []
    sse = n = 0
    for o, i in zip(obj_points, img_points):
        img = np.asarray(i, np.float32).reshape(-1, 2)
        obj = np.asarray(o, np.float32).reshape(-1, 3)
        if len(img) < 4:
            continue
        ok, r, t = cv2.solvePnP(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2),
                                K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        proj = cv2.projectPoints(obj, r, t, K, dist)[0].reshape(-1, 2)
        if not np.all(np.isfinite(proj)):
            continue
        d = proj.astype(np.float64) - img.astype(np.float64)
        sse += float((d * d).sum()); n += d.shape[0]
        rvecs.append(r); tvecs.append(t); filt_obj.append(obj); filt_img.append(img)
    rmse = float(np.sqrt(sse / n)) if n else float("nan")
    return rmse, rvecs, tvecs, filt_obj, filt_img


# ---- refractive bundle adjustment --------------------------------------------

def _rodrigues(rvec):
    """Differentiable axis-angle → rotation matrix."""
    theta = torch.linalg.norm(rvec) + 1e-12
    k = rvec / theta
    Kx = torch.zeros(3, 3, dtype=rvec.dtype, device=rvec.device)
    Kx[0, 1] = -k[2]; Kx[0, 2] = k[1]
    Kx[1, 0] = k[2];  Kx[1, 2] = -k[0]
    Kx[2, 0] = -k[1]; Kx[2, 1] = k[0]
    I = torch.eye(3, dtype=rvec.dtype, device=rvec.device)
    return I + torch.sin(theta) * Kx + (1 - torch.cos(theta)) * (Kx @ Kx)


def _project_refractive(obj, rvec, tvec, n, K, D):
    """Full physical water model: project obj → pixel through refraction + lens."""
    R = _rodrigues(rvec)
    X = obj @ R.T + tvec
    x = X[:, 0] / X[:, 2]
    y = X[:, 1] / X[:, 2]
    r2 = x * x + y * y
    # Snell-derived forward refraction (Eq. 8):  m = n / sqrt(1 + r² − n²r²)
    m = n / torch.sqrt(torch.clamp(1 + r2 - n * n * r2, min=1e-8))
    xr, yr = m * x, m * y
    # Brown–Conrady forward lens distortion with the fixed air D.
    k1, k2, p1, p2, k3 = D
    rr2 = xr * xr + yr * yr
    radial = 1 + k1 * rr2 + k2 * rr2 ** 2 + k3 * rr2 ** 3
    xd = xr * radial + 2 * p1 * xr * yr + p2 * (rr2 + 2 * xr * xr)
    yd = yr * radial + p1 * (rr2 + 2 * yr * yr) + 2 * p2 * xr * yr
    u = K[0, 0] * xd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]
    return torch.stack([u, v], dim=1)


def refractive_bundle_adjustment(obj_lists, img_lists, air_K, air_D,
                                 rvec_init, tvec_init, n_init=1.33,
                                 epochs=5000, lr=1e-3, val_ratio=0.2, seed=0):
    """Jointly fit (n, {rvec, tvec}) under the Singh & Alexis 2024 model.

    Pixel-space bundle adjustment with `air_K`, `air_D` fixed (they're
    medium-independent). Per-view points are split 80/20 into train and
    validation subsets; only the train subset drives the loss, and both
    train and val RMSE are recorded per epoch.
    """
    rng = np.random.default_rng(seed)

    K = torch.as_tensor(np.asarray(air_K), dtype=DTYPE, device=DEVICE)
    D_arr = np.zeros(5, dtype=np.float64)
    D_in = np.asarray(air_D).reshape(-1)
    D_arr[: min(5, len(D_in))] = D_in[: min(5, len(D_in))]
    D = torch.as_tensor(D_arr, dtype=DTYPE, device=DEVICE)

    raw_n0 = float(np.log(np.expm1(max(n_init - 1.0, 1e-6))))
    raw_n = torch.tensor(raw_n0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rvec_params = [torch.tensor(np.asarray(r).reshape(3), dtype=DTYPE,
                                device=DEVICE, requires_grad=True) for r in rvec_init]
    tvec_params = [torch.tensor(np.asarray(t).reshape(3), dtype=DTYPE,
                                device=DEVICE, requires_grad=True) for t in tvec_init]

    # Tensorize points + build per-view train/val index masks.
    obj_t, img_t, tr_idx, vl_idx = [], [], [], []
    for obj, img in zip(obj_lists, img_lists):
        obj = np.asarray(obj).reshape(-1, 3)
        img = np.asarray(img).reshape(-1, 2)
        m = len(img)
        perm = rng.permutation(m)
        n_tr = max(1, int(m * (1 - val_ratio)))
        obj_t.append(torch.as_tensor(obj, dtype=DTYPE, device=DEVICE))
        img_t.append(torch.as_tensor(img, dtype=DTYPE, device=DEVICE))
        tr_idx.append(torch.as_tensor(perm[:n_tr], device=DEVICE))
        vl_idx.append(torch.as_tensor(perm[n_tr:], device=DEVICE))

    opt = optim.Adam([raw_n] + rvec_params + tvec_params, lr=lr)
    train_rmse, val_rmse = [], []
    n_train = sum(int(i.numel()) for i in tr_idx)
    n_val = sum(int(i.numel()) for i in vl_idx)

    for epoch in range(epochs):
        opt.zero_grad()
        n = 1.0 + F.softplus(raw_n)

        sse_tr = torch.zeros((), dtype=DTYPE, device=DEVICE)
        sse_vl = torch.zeros((), dtype=DTYPE, device=DEVICE)
        for obj, img, r, t, ti, vi in zip(obj_t, img_t, rvec_params, tvec_params, tr_idx, vl_idx):
            proj = _project_refractive(obj, r, t, n, K, D)
            sse_tr = sse_tr + ((proj[ti] - img[ti]) ** 2).sum()
            if vi.numel() > 0:
                with torch.no_grad():
                    sse_vl = sse_vl + ((proj[vi].detach() - img[vi]) ** 2).sum()

        loss = sse_tr / (2 * n_train)  # /2 because each point contributes u²+v²
        loss.backward()
        opt.step()

        train_rmse.append(float(torch.sqrt(sse_tr.detach() / n_train).item()))
        val_rmse.append(float(torch.sqrt(sse_vl / max(n_val, 1)).item()) if n_val else float("nan"))
        if epoch % 500 == 0:
            print(f"  BA epoch {epoch}: train_rmse={train_rmse[-1]:.4f}px "
                  f"val_rmse={val_rmse[-1]:.4f}px n={float(n):.4f}")

    n_final = float(1.0 + F.softplus(raw_n.detach()).cpu())
    rvecs_out = [r.detach().cpu().numpy().reshape(3, 1) for r in rvec_params]
    tvecs_out = [t.detach().cpu().numpy().reshape(3, 1) for t in tvec_params]
    print(f"  final: n = {n_final:.6f}  train_rmse = {train_rmse[-1]:.4f}px  "
          f"val_rmse = {val_rmse[-1]:.4f}px")
    return {
        "n": n_final,
        "rvecs": rvecs_out, "tvecs": tvecs_out,
        "train_rmse": train_rmse, "val_rmse": val_rmse,
    }


# ---- pipelines ---------------------------------------------------------------

def _plot_ba_loss(ba, m1_rmse, m2_rmse, n, save_path, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ba["train_rmse"], label="train", color="steelblue")
    ax.plot(ba["val_rmse"], label="val", color="darkorange")
    ax.axhline(m1_rmse, color="red", linestyle="--", label=f"air_direct ({m1_rmse:.2f})")
    ax.axhline(m2_rmse, color="green", linestyle="--", label=f"water_opencv ({m2_rmse:.2f})")
    ax.set_xlabel("BA epoch"); ax.set_ylabel("RMSE (px)")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title(f"{title}  (n = {n:.4f})")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close(fig)


def calibrate_water_from_air(water_points_file, air_opencv_path, output_dir):
    """Compare three methods on water points given an air calibration:
      1. Direct use of air intrinsics (PnP + reproject).
      2. Full OpenCV recalibration on water points.
      3. Refractive bundle adjustment (Singh & Alexis 2024) with air K/D fixed.
    """
    data = np.load(water_points_file, allow_pickle=True)
    obj_points = [a.astype(np.float32) for a in data["obj_points"]]
    img_points = [a.astype(np.float32) for a in data["img_points"]]
    img_size = tuple(int(x) for x in data["img_size"])
    cam = Path(water_points_file).stem

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    air = load_opencv_result(air_opencv_path)
    air_K, air_D = air["K"], air["dist"]

    # Method 1: air intrinsics, PnP on water.
    m1_rmse, m1_rvecs, m1_tvecs, _, _ = _direct_reprojection_rmse(
        obj_points, img_points, air_K, air_D)
    np.savez(str(out / f"{cam}_air_direct.npz"),
             K=air_K, dist=air_D,
             rvecs=np.array(m1_rvecs, dtype=object),
             tvecs=np.array(m1_tvecs, dtype=object),
             img_size=np.array(img_size), rmse=m1_rmse)

    # Method 2: recalibrate K/D on water points with a per-view 80/20 split.
    # The calibration sees only the training points; the val RMSE reports how
    # well the fitted (K, D, poses) generalize to the held-out 20%.
    tr_obj, tr_img, vl_obj, vl_img = _split_points(obj_points, img_points)
    m2 = opencv_full_calib(tr_obj, tr_img, img_size)
    # opencv_full_calib may drop degenerate views from the TRAIN subset; align
    # val lists to the same kept views.
    vl_obj_kept = [vl_obj[i] for i in m2["kept_indices"]]
    vl_img_kept = [vl_img[i] for i in m2["kept_indices"]]
    m2_train_rmse = m2["rmse_px"]
    m2_val_rmse = _reprojection_rmse(
        vl_obj_kept, vl_img_kept, m2["K"], m2["dist"], m2["rvecs"], m2["tvecs"])
    np.savez(str(out / f"{cam}_water_opencv.npz"),
             K=m2["K"], dist=m2["dist"],
             rvecs=np.array(m2["rvecs"], dtype=object),
             tvecs=np.array(m2["tvecs"], dtype=object),
             filtered_obj=np.array(m2["filtered_obj"], dtype=object),
             filtered_img=np.array(m2["filtered_img"], dtype=object),
             img_size=np.array(img_size),
             train_rmse=m2_train_rmse, val_rmse=m2_val_rmse)

    # Method 3: refractive bundle adjustment.
    ba = refractive_bundle_adjustment(
        m2["filtered_obj"], m2["filtered_img"], air_K, air_D,
        rvec_init=m2["rvecs"], tvec_init=m2["tvecs"])
    np.savez(str(out / f"{cam}_refractive.npz"),
             K=air_K, dist=air_D, n=ba["n"],
             rvecs=np.array(ba["rvecs"], dtype=object),
             tvecs=np.array(ba["tvecs"], dtype=object),
             img_size=np.array(img_size),
             train_rmse=np.array(ba["train_rmse"]),
             val_rmse=np.array(ba["val_rmse"]))
    _plot_ba_loss(ba, m1_rmse, m2_train_rmse, ba["n"],
                  out / f"{cam}_refractive_loss.png",
                  f"{cam} refractive BA")

    summary = {
        "air_direct": m1_rmse,
        "water_opencv_train": m2_train_rmse,
        "water_opencv_val": m2_val_rmse,
        "refractive_train": ba["train_rmse"][-1],
        "refractive_val": ba["val_rmse"][-1],
    }
    print(f"\n{cam} cross-medium comparison:")
    for k, v in summary.items():
        print(f"  {k:20s} RMSE = {v:.6f} px")
    print(f"  learned refractive index n = {ba['n']:.6f} (water ≈ 1.33)")
    summary["learned_n"] = ba["n"]
    return summary


if __name__ == "__main__":
    air_output_dir = Path("../calibration/air")
    water_points_dir = Path("../points/water")
    water_output_dir = Path("../calibration/water")

    print(f"Using device: {DEVICE}  dtype: {DTYPE}")

    for npz in sorted(water_points_dir.glob("cam*.npz")):
        cam = npz.stem
        air_path = air_output_dir / f"{cam}_opencv.npz"
        if not air_path.exists():
            print(f"Skipping {cam}: no air calibration at {air_path}")
            continue
        calibrate_water_from_air(str(npz), str(air_path), str(water_output_dir))
