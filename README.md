# Refractive Bundle Adjustment for Underwater Camera Calibration

This module calibrates a flat-port underwater camera **without running a
full calibration target underwater**. Given a camera already calibrated in
air, it recovers the refractive index $n$ of the medium (e.g. water)
together with the per-view target poses by joint bundle adjustment. The
physical model follows Singh & Alexis, *Online Refractive Camera Model
Calibration in Visual Inertial Odometry* (ICRA 2024).

---

## 1. Projection Model

Let $X \in \mathbb{R}^{3}$ be a 3D point expressed in the camera frame
after applying the pose $(R, t)$. Its air-equivalent normalized ray
direction is

$$
\bar{p}_{\mathcal{C}} \;=\; \left( \tfrac{X_x}{X_z},\; \tfrac{X_y}{X_z} \right), \qquad r = \|\bar{p}_{\mathcal{C}}\|.
$$

Looking from a flat glass port, Snell's law gives $\sin\theta_{\alpha}
= n\sin\theta_{\mu}$. In normalized coordinates this yields a radial
scaling of the ray before it reaches the lens (paper Eq. 8):

$$
\bar{p}_{\mathcal{C},r} \;=\; m(n, r)\,\bar{p}_{\mathcal{C}},
\qquad
m(n, r) \;=\; \dfrac{n}{\sqrt{\,1 + r^{2} - n^{2} r^{2}\,}}.
$$

The distorted ray $\bar{p}_{\mathcal{C},r}$ then passes through the
(medium-independent) air lens distortion $D_{\text{air}}$ and the air
intrinsic matrix $K_{\text{air}}$ to become a pixel

$$
u \;=\; K_{\text{air}} \cdot D_{\text{air}}\!\bigl(\bar{p}_{\mathcal{C},r}\bigr).
$$

We package this whole chain as $\pi(n, R, t, X)$.

### Inverse (used for PnP at inference)

Given an observed water pixel $u$, we first lens-undistort with
$(K_{\text{air}}, D_{\text{air}})$ to recover $\bar{p}_{\mathcal{C},r}$,
then invert the Snell scaling (paper Eq. 9):

$$
\bar{p}_{\mathcal{C}} \;=\; \dfrac{\bar{p}_{\mathcal{C},r}}{\sqrt{\,n^{2}r_r^{2} + n^{2} - r_r^{2}\,}},
\qquad
r_r = \|\bar{p}_{\mathcal{C},r}\|.
$$

Re-pixelating with $K_{\text{air}}$ and running `solvePnP` against zero
distortion yields the pose of a target under the refractive model.

---

## 2. The Bundle Adjustment Problem

Fix $K_{\text{air}}, D_{\text{air}}$ to the air calibration. The unknowns
are:

- one scalar $n$, shared across all views,
- one pose $(R_i, t_i)$ per view $i$.

Given $N$ views with observed pixels $\{u_{i,j}\}$ of known 3D target
points $\{X_{i,j}\}$, we solve

$$
\boxed{\;
\min_{\,n,\,\{R_i, t_i\}} \;\; \sum_{i,j} \bigl\| u_{i,j} - \pi(n, R_i, t_i, X_{i,j}) \bigr\|^{2}
\;}
$$

subject to $n > 1$ (enforced by reparametrizing $n = 1 + \text{softplus}(\tilde{n})$).

The training split randomly removes 20 % of the points in each view as a
validation set; they feed through the same poses but do not contribute
to the gradient, so the validation RMSE reports generalization to points
the optimizer never saw.

---

## 3. Why Joint Optimization Matters

A simpler workflow — run `cv2.calibrateCamera` on water points to recover
$(K_{\text{water}}, D_{\text{water}}, \{R_i^{\mathrm{ocv}}, t_i^{\mathrm{ocv}}\})$,
freeze the poses, then fit $n$ alone — **does not work**. The OpenCV
calibration absorbs the Snell scaling into a shifted effective focal
length, biasing the poses. Holding those poses fixed forces $n$ to
compensate for pose error, and the optimizer converges to the wrong
value (in our data, $n \approx 2.5$).

Letting both $n$ and the poses move together breaks this ambiguity:
Snell refraction is the only place in the pipeline that introduces
radially varying magnification, so the optimum recovers the physical
$n \approx 1.33$.

---

## 4. Algorithm (Pseudocode)

```
Input:
    {X_i, U_i}             per-view 3D points and observed pixels
    K_air, D_air           fixed air intrinsics
    R_i^0, t_i^0           initial poses (warm start from water OpenCV calib)
    n_0                    initial refractive index (default 1.33)
    α, E                   learning rate, number of epochs
    ρ                      validation ratio (default 0.2)

Output:
    n*, {R_i*, t_i*}       learned refractive index and refined poses

# Parameters — reparametrize n so n > 1 is hard-enforced
ñ    ← log(exp(n_0 − 1) − 1)                     # inverse of softplus
r_i  ← rvec(R_i^0)                                 # Rodrigues axis-angle
t_i  ← t_i^0
mark (ñ, {r_i}, {t_i}) as requires_grad

# Per-view train/val split
for each view i:
    π_i ← random permutation of the points in view i
    T_i ← π_i[: ⌈(1 − ρ) · |view_i|⌉]            # train indices
    V_i ← π_i[⌈(1 − ρ) · |view_i|⌉ :]            # val indices

# Optimize
opt ← Adam({ñ} ∪ {r_i} ∪ {t_i}, lr = α)
for epoch e = 1 … E:
    n ← 1 + softplus(ñ)
    sse_tr ← 0 ; sse_vl ← 0
    for each view i:
        R_i ← rodrigues(r_i)
        for each point j in view i:
            X ← R_i · X_{i,j} + t_i
            p̄  ← (X_x / X_z, X_y / X_z)
            r² ← ‖p̄‖²
            m  ← n / √(1 + r² − n² r²)           # Snell scaling
            p̄_r ← m · p̄
            p̄_d ← brown_conrady(p̄_r, D_air)
            û_{i,j} ← K_air · (p̄_d, 1)
            if j ∈ T_i: sse_tr += ‖u_{i,j} − û_{i,j}‖²
            if j ∈ V_i: sse_vl += ‖u_{i,j} − û_{i,j}‖²   # no grad
    loss ← sse_tr / (2 · |⋃ T_i|)                 # MSE per coordinate
    loss.backward()
    opt.step()
    record √(sse_tr / |⋃ T_i|), √(sse_vl / |⋃ V_i|)

return  n*, {R_i*, t_i*}
```

In this repository the algorithm is implemented in
[`intrinsic/intrinsic_calibration.py`](intrinsic/intrinsic_calibration.py):

| Symbol in the math                              | Where it lives                         |
| ----------------------------------------------- | -------------------------------------- |
| $\pi(\cdot)$                                    | `_project_refractive`                  |
| Rodrigues $r_i \to R_i$                         | `_rodrigues`                           |
| softplus-reparametrized $n$                     | `n = 1 + F.softplus(raw_n)`            |
| joint optimization loop                         | `refractive_bundle_adjustment`         |
| inverse model (for PnP at inference)            | `solve_pnp_refractive` in `localization/localization.py` |

---

## 5. Inputs, Outputs, Hyperparameters

**Inputs.** Per-view object points $X_{i,j}$ and detected pixels
$u_{i,j}$ (e.g. from a ChArUco board), the image size, the air
calibration `{K, dist}`, and warm-start poses — in practice obtained
from `cv2.calibrateCamera` run on the same water points.

**Outputs.** The learned refractive index $n$, refined per-view poses
$(R_i, t_i)$, and train / val RMSE histories, saved as
`{cam}_refractive.npz`.

**Hyperparameters (defaults).**
$E = 5000$ epochs, $\alpha = 10^{-3}$, $\rho = 0.2$, $n_0 = 1.33$. In
practice convergence is monotonic and the val curve tracks the train
curve (no overfitting — there is effectively one scalar to overfit
with).

**Device.** CUDA + `float32` when available, otherwise CPU + `float64`.
Picked automatically at the top of `intrinsic_calibration.py`.
