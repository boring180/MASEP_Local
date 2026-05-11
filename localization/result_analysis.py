"""Analyse localization results: compare intrinsic methods, plots, stats."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def filter_best_frames(results, keep=0.99):
    """Drop the worst (1 − keep) fraction of frames by inter-camera spread.

    For each multi-camera frame, compute the sum of per-axis stdev across
    detecting cameras. Frames above the `keep`-quantile of that score are
    replaced with an empty dict (i.e. treated as un-localized downstream).
    Frames with fewer than 2 detecting cameras have no spread signal and
    are always kept.
    """
    spreads = []
    for i, frame in enumerate(results):
        if len(frame) < 2:
            continue
        pts = np.array([T[:3, 3] for T in frame.values()])
        spreads.append((i, float(pts.std(axis=0).sum())))
    if not spreads:
        return list(results)
    threshold = np.quantile([s for _, s in spreads], keep)
    drop = {i for i, s in spreads if s > threshold}
    return [{} if i in drop else f for i, f in enumerate(results)]


def extract_positions(results):
    """Extract board XYZ positions per frame (average across detecting cameras).

    Args:
        results: list of dicts from process_video, each {cam: 4x4 world pose}

    Returns:
        list of (N,3) arrays or None per frame
    """
    positions = []
    for frame in results:
        if not frame:
            positions.append(None)
            continue
        pts = [T[:3, 3] for T in frame.values()]
        positions.append(np.mean(pts, axis=0))
    return positions


def extract_per_camera_positions(results):
    """Extract per-camera board positions per frame.

    Returns:
        dict of {cam: list of (xyz or None) per frame}
    """
    cams = set()
    for frame in results:
        cams.update(frame.keys())

    per_cam = {cam: [] for cam in cams}
    for frame in results:
        for cam in cams:
            if cam in frame:
                per_cam[cam].append(frame[cam][:3, 3])
            else:
                per_cam[cam].append(None)
    return per_cam


def compare_methods(all_positions):
    """Print statistics for each method."""
    print(f"\n{'Method':<10} {'Frames':>8} {'Mean X':>10} {'Mean Y':>10} {'Mean Z':>10} "
          f"{'Std X':>10} {'Std Y':>10} {'Std Z':>10}")
    print("-" * 78)
    for method, positions in all_positions.items():
        valid = np.array([p for p in positions if p is not None])
        if len(valid) == 0:
            print(f"{method:<10} {'0':>8}")
            continue
        mean = valid.mean(axis=0)
        std = valid.std(axis=0)
        print(f"{method:<10} {len(valid):>8} {mean[0]:>10.4f} {mean[1]:>10.4f} {mean[2]:>10.4f} "
              f"{std[0]:>10.4f} {std[1]:>10.4f} {std[2]:>10.4f}")


def plot_trajectories(all_positions, save_path="localization_comparison.png"):
    """Plot XYZ trajectories for each method."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    labels = ["X", "Y", "Z"]

    for method, positions in all_positions.items():
        frames, xyz = [], [[], [], []]
        for i, p in enumerate(positions):
            if p is not None:
                frames.append(i)
                for j in range(3):
                    xyz[j].append(p[j])
        for j in range(3):
            axes[j].plot(frames, xyz[j], label=method, marker='.', markersize=2, linewidth=0.8)

    for j in range(3):
        axes[j].set_ylabel(f"{labels[j]} (m)")
        axes[j].legend()
        axes[j].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame")
    fig.suptitle("Charuco Localization Under Water - Method Comparison")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_pairwise_diff(all_positions, ref_method="opencv", save_path="localization_diff.png"):
    """Plot position difference between each method and the reference."""
    ref = all_positions[ref_method]
    other_methods = [m for m in all_positions if m != ref_method]

    fig, axes = plt.subplots(len(other_methods), 1,
                             figsize=(14, 4 * len(other_methods)), sharex=True)
    if len(other_methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, other_methods):
        frames, diffs = [], []
        pos = all_positions[method]
        for i in range(len(ref)):
            if ref[i] is not None and i < len(pos) and pos[i] is not None:
                frames.append(i)
                diffs.append(np.linalg.norm(pos[i] - ref[i]))
        ax.plot(frames, diffs, marker='.', markersize=2, linewidth=0.8)
        ax.set_ylabel(f"|{method} - {ref_method}| (m)")
        ax.grid(True, alpha=0.3)
        if diffs:
            ax.set_title(f"{method} vs {ref_method}: mean={np.mean(diffs):.4f}m, max={np.max(diffs):.4f}m")

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_inter_camera_mse(all_results, save_path="localization_mse.png"):
    """Render a matplotlib table of MSE between camera pairs for each method.

    Args:
        all_results: dict of {method: list of frame dicts {cam: 4x4 pose}}
    """
    from itertools import combinations

    methods = list(all_results.keys())
    all_cams = set()
    for results in all_results.values():
        for frame in results:
            all_cams.update(frame.keys())
    cam_pairs = sorted(combinations(sorted(all_cams), 2))

    if not cam_pairs:
        return

    col_labels = [f"{c1} - {c2}" for c1, c2 in cam_pairs]
    cell_text = []
    for method in methods:
        results = all_results[method]
        row = []
        for c1, c2 in cam_pairs:
            sq_errors = []
            for frame in results:
                if c1 in frame and c2 in frame:
                    diff = frame[c1][:3, 3] - frame[c2][:3, 3]
                    sq_errors.append(np.sum(diff ** 2))
            row.append(f"{np.mean(sq_errors):.6f}" if sq_errors else "N/A")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(3 + 2 * len(cam_pairs), 0.6 + 0.5 * len(methods)))
    ax.axis('off')
    table = ax.table(cellText=cell_text, rowLabels=methods, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    ax.set_title("Inter-Camera Localization MSE (m²)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_scatter_grid(all_results, save_path="localization_scatter.png"):
    """3x3 scatter grid: rows = methods, columns = X/Y/Z, colored by camera.

    Args:
        all_results: dict of {method: list of frame dicts {cam: 4x4 pose}}
    """
    methods = list(all_results.keys())
    coord_labels = ["X", "Y", "Z"]
    cam_colors = {"cam2": "red", "cam1": "green", "cam0": "blue"}

    fig, axes = plt.subplots(len(methods), 3, figsize=(12, 3.5 * len(methods)))
    if len(methods) == 1:
        axes = axes[np.newaxis, :]

    for row, method in enumerate(methods):
        results = all_results[method]
        per_cam = extract_per_camera_positions(results)

        for col in range(3):  # X, Y, Z
            ax = axes[row, col]
            for cam in sorted(per_cam.keys()):
                frames, vals = [], []
                for i, p in enumerate(per_cam[cam]):
                    if p is not None:
                        frames.append(i)
                        vals.append(p[col])
                ax.scatter(frames, vals, s=6, alpha=0.6, label=cam,
                           color=cam_colors.get(cam, 'gray'))

            ax.set_ylabel(f"{coord_labels[col]} (m)")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(coord_labels[col])
            if row == len(methods) - 1:
                ax.set_xlabel("Frame")
            if col == 0:
                ax.annotate(method, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 10, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            fontsize=12, fontweight='bold', ha='right', va='center')
            if row == 0 and col == 2:
                ax.legend(markerscale=3, fontsize=8)

    fig.suptitle("Per-Camera Localization Scatter", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
