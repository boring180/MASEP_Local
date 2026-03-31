"""
Multi-camera ArUco marker localization via PnP.

Provides:
  - Capture : video recording with multi-camera frame handling
  - Localization(Capture) : ArUco detection + PnP pose → world coordinates
"""

import cv2
import json
import numpy as np
import os
import pickle
from datetime import datetime
from pathlib import Path
import tqdm


class Capture:
    def __init__(self, setting_path="setting.json", cameras=None):
        with open(setting_path, "r") as f:
            self.settings = json.load(f)

        self.cameras = cameras
        if cameras is not None:
            self.reference_shape = cameras[0].read()[1].shape[:2]
        else:
            self.reference_shape = None

    def __del__(self):
        if self.cameras is not None:
            for cam in self.cameras:
                cam.release()

    # ---- frame layout helpers ---- #
    def _slice_frame(self, frame):
        n = len(self.settings["cameras"])
        h = frame.shape[0] // n
        return [frame[i * h : (i + 1) * h, :] for i in range(n)]

    def _concat_frames(self, frames):
        ref = self.reference_shape
        resized = [cv2.resize(f, (ref[1], ref[0])) for f in frames]
        return np.concatenate(resized, axis=0)

    # ---- serialise one frame's detections to JSON-friendly dict ---- #
    @staticmethod
    def _frame_to_json(frame_data):
        frame_json = {}
        for cam_name, detections in frame_data.items():
            if isinstance(detections, dict):
                items = []
                for marker_id, coord in detections.items():
                    arr = np.array(coord).reshape(-1)
                    if arr.size >= 3:
                        items.append({"id": int(marker_id),
                                      "x": float(arr[0]), "y": float(arr[1]), "z": float(arr[2])})
                frame_json[cam_name] = items
            else:
                frame_json[cam_name] = []
        return frame_json

    # ---- capture from video file ---- #
    def reproduce_capture(self, capture_fn, video_path):
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reference_shape = (h // len(self.settings["cameras"]), w)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs("output", exist_ok=True)
        stem = Path(video_path).stem
        out = cv2.VideoWriter(f"output/{stem}_reproduce.mp4",
                              cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
        data = []
        for _ in tqdm.tqdm(range(total)):
            ret, frame = cap.read()
            if not ret:
                break
            frames = self._slice_frame(frame)
            fd = {self.settings["cameras"][i]: capture_fn(frames[i], self.settings["cameras"][i])
                  for i in range(len(self.settings["cameras"]))}
            data.append(fd)
            out.write(self._concat_frames(frames))

        with open(f"output/{stem}_reproduce.json", "w") as f:
            json.dump([self._frame_to_json(d) for d in data], f, indent=2)
        out.release()
        cap.release()

    # ---- live capture ---- #
    def save_video(self, capture_fn=None, save_preview=False):
        if capture_fn is None:
            capture_fn = lambda frame, name: {}

        for cam in self.cameras:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        ret, first = self.cameras[0].read()
        self.reference_shape = first.shape[:2]

        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cams = self.settings["cameras"]
        fname = f"output/{cams[0]}_{timestamp}.mp4" if len(cams) == 1 else f"output/{timestamp}.mp4"

        sample = self._concat_frames([first] * len(self.cameras))
        out = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*"mp4v"), 24,
                              (sample.shape[1], sample.shape[0]))
        data = []
        while True:
            frames, show_frames, fd = [], [], {}
            for i, cam in enumerate(self.cameras):
                ret, frame = cam.read()
                frames.append(frame)
                shown = frame.copy() if not save_preview else frame
                fd[cams[i]] = capture_fn(shown, cams[i])
                show_frames.append(shown)

            out.write(self._concat_frames(frames))
            cv2.imshow("Frames", self._concat_frames(show_frames))
            data.append(fd)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        with open(f"output/{timestamp}.json", "w") as f:
            json.dump([self._frame_to_json(d) for d in data], f, indent=2)
        out.release()
        cv2.destroyAllWindows()


class Localization(Capture):
    """ArUco marker detection + PnP pose estimation using calibrated cameras."""

    def __init__(self, setting_path="setting.json", cameras=None):
        super().__init__(setting_path, cameras)

        param_dir = self.settings["camera_parameter_path"]
        self.cameras_mtx, self.cameras_dist, self.cameras_extrinsic = {}, {}, {}
        for cam in self.settings["cameras"]:
            with open(f"{param_dir}/mtx_{cam}.pkl", "rb") as f:
                self.cameras_mtx[cam] = pickle.load(f)
            with open(f"{param_dir}/dist_{cam}.pkl", "rb") as f:
                self.cameras_dist[cam] = pickle.load(f)
            with open(f"{param_dir}/extrinsic_{cam}.pkl", "rb") as f:
                self.cameras_extrinsic[cam] = pickle.load(f)

        dict_name = self.settings["aruco_dict_localization"]
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    def localization(self, frame, camera_name):
        """Detect ArUco markers and estimate their world position via PnP + extrinsic."""
        corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        results = {}
        if ids is None:
            return results

        mtx = self.cameras_mtx[camera_name]
        dist = self.cameras_dist[camera_name]
        ext = self.cameras_extrinsic[camera_name]
        marker_len = self.settings["marker_size_localization"]

        for i in range(len(corners)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], marker_len, mtx, dist
            )
            T_marker = np.eye(4)
            T_marker[:3, :3], _ = cv2.Rodrigues(rvec)
            T_marker[:3, 3] = tvec.flatten()

            world_T = ext @ T_marker
            pos = world_T[:3, 3]
            results[ids[i][0]] = pos

            label = f"ID:{ids[i][0]} X:{pos[0]:.2f} Y:{pos[1]:.2f} Z:{pos[2]:.2f}"
            org = (int(corners[i][0][0][0]), int(corners[i][0][0][1]))
            cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return results

    def detection(self, frame, camera_name):
        """Detect ArUco markers (2-D corners only, no pose)."""
        corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        results = {}
        if ids is None:
            return results
        for i in range(len(corners)):
            results[ids[i][0]] = corners[i]
            org = (int(corners[i][0][0][0]), int(corners[i][0][0][1]))
            cv2.putText(frame, f"ID:{ids[i][0]}", org,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return results


if __name__ == "__main__":
    loc = Localization(cameras=[cv2.VideoCapture(2), cv2.VideoCapture(1), cv2.VideoCapture(0)])
    loc.save_video(loc.localization, save_preview=True)
