"""Analyse localization JSON output: inter-camera differences, plots, animation."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgb


class ResultAnalysis:
    def __init__(self, json_path, setting_path="setting.json"):
        with open(setting_path, "r") as f:
            self.settings = json.load(f)
        with open(json_path, "r") as f:
            self.data = json.load(f)
        os.makedirs("result", exist_ok=True)

    @staticmethod
    def _coord(detection):
        if "position" in detection:
            return detection["position"]
        return [detection["x"], detection["y"], detection["z"]]

    def extract_position(self, cam1, cam2=None):
        pts1, pts2 = [], []
        for frame in self.data:
            if len(frame.get(cam1, [])) == 0:
                continue
            if cam2 is not None and len(frame.get(cam2, [])) == 0:
                continue
            pts1.append(self._coord(frame[cam1][0]))
            if cam2 is not None:
                pts2.append(self._coord(frame[cam2][0]))
        return np.array(pts1), np.array(pts2)

    def inter_camera_difference(self, camera_name):
        cam_pts, center_pts = self.extract_position(camera_name, self.settings["center_camera"])
        diff = cam_pts - center_pts
        msg = f"{camera_name} vs {self.settings['center_camera']}  mean={diff.mean(0)}  std={diff.std(0)}"
        print(msg)
        with open("result/difference.log", "a") as f:
            f.write(msg + "\n")

    def visualize_points(self, camera_names=None, save=True):
        """Plot 2-D projections for one or more cameras on the same axes."""
        if camera_names is None:
            camera_names = self.settings["cameras"]
        if isinstance(camera_names, str):
            camera_names = [camera_names]

        labels = ["X", "Y", "Z"]
        colors = ["red", "green", "blue", "orange", "purple"]
        cam_pts = {c: self.extract_position(c)[0] for c in camera_names}

        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        for ax, drop in zip(axes, range(3)):
            keep = [j for j in range(3) if j != drop]
            for ci, c in enumerate(camera_names):
                if len(cam_pts[c]) == 0:
                    continue
                ax.scatter(cam_pts[c][:, keep[0]], cam_pts[c][:, keep[1]],
                           color=colors[ci % len(colors)], alpha=0.5, s=8, label=c)
            ax.set_xlabel(labels[keep[0]])
            ax.set_ylabel(labels[keep[1]])
            ax.set_aspect("equal")
            if len(camera_names) > 1:
                ax.legend(fontsize=7)
        plt.tight_layout()
        if save:
            suffix = "_".join(camera_names) if len(camera_names) <= 3 else "all"
            plt.savefig(f"result/points_{suffix}.png", dpi=150)
        plt.close(fig)

    def animate(self, save_path="result/animation.mp4", trail=100):
        cams = self.settings["cameras"]
        colors = ["red", "green", "blue", "orange", "purple"]

        all_pts = np.concatenate([self.extract_position(c)[0] for c in cams
                                  if len(self.extract_position(c)[0]) > 0])
        pad = (all_pts.max(0) - all_pts.min(0)) * 0.1
        lo, hi = all_pts.min(0) - pad, all_pts.max(0) + pad

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_aspect("equal")

        scatters = {}
        history = {c: [] for c in cams}
        for ci, c in enumerate(cams):
            scatters[c] = ax.scatter([], [], color=colors[ci % len(colors)], label=c, s=20)
        ax.legend()

        def update(fi):
            fd = self.data[fi]
            artists = []
            for ci, c in enumerate(cams):
                if c in fd and len(fd[c]) > 0:
                    history[c].append(self._coord(fd[c][0]))
                    if len(history[c]) > trail:
                        history[c].pop(0)
                if history[c]:
                    pts = np.array(history[c])[:, :2]
                    rgba = np.zeros((len(pts), 4))
                    rgba[:, :3] = to_rgb(colors[ci % len(colors)])
                    rgba[:, 3] = np.linspace(0.01, 1, len(pts))
                    scatters[c].set_offsets(pts)
                    scatters[c].set_color(rgba)
                else:
                    scatters[c].set_offsets(np.empty((0, 2)))
                artists.append(scatters[c])
            return artists

        ani = FuncAnimation(fig, update, frames=len(self.data), interval=50, blit=True)
        ani.save(save_path, writer="ffmpeg", fps=20)
        plt.close(fig)
        print(f"Animation saved to {save_path}")


if __name__ == "__main__":
    ra = ResultAnalysis("output/20251116_160147.json")
    for cam in ra.settings["cameras"]:
        ra.inter_camera_difference(cam)
        ra.visualize_points(cam)
    ra.visualize_points()
    ra.animate()
