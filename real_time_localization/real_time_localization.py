import cv2
import json
import numpy as np
import os
import pickle
from datetime import datetime

class Capture:
    def __init__(self, cameras):
        try:
            with open('setting.json', 'r') as f:
                self.settings = json.load(f)
        except:
            raise Exception(f"Error loading settings from setting.json")
        
        self.cameras = cameras
        self.reference_shape = self.cameras[0].read()[1].shape[:2]
        

    def __del__(self):
        for i in range(len(self.cameras)):
            self.cameras[i].release()
            
    def __str__(self):
        return json.dumps(self.settings, indent=4)
            
    def default_capture(self, frame, camera_name):
        return frame
    
    def chessboard_capture(self, frame, camera_name):
        row_number = self.settings['pattern_size'][0]
        col_number = self.settings['pattern_size'][1]
        internal_corners_X = col_number - 1
        internal_corners_Y = row_number - 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (internal_corners_Y, internal_corners_X), None, getattr(cv2, self.settings['chessboard_flags']))
        if ret:
            text = "Detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 15
            thickness = 30
            color = (0, 0, 255)
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            center_x = frame.shape[1] // 2
            center_y = frame.shape[0] // 2
            org = (center_x - text_width // 2, center_y + text_height // 2)
            cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
        return corners
        
    def charuco_capture(self, frame, camera_name):
        pass
    
    def save_video(self, capture_function = None, save_preview = False):
        if capture_function is None:
            capture_function = self.default_capture
            
        frames = []
            
        for i in range(len(self.cameras)):
            self.cameras[i].set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cameras[i].set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
            if not self.cameras[i].isOpened():
                print(f"Error: Could not open camera {i}")
                return
            ret, frame = self.cameras[i].read()
            frames.append(frame)
        
        frame = self.frame_concatent(frames, self.reference_shape)
        self.height, self.width = frame.shape[:2]
        
        if not os.path.exists('output'):
            os.makedirs('output')
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'output/{timestamp}.mp4'
        
        if len(self.settings['cameras']) == 1:
            filename = f"output/{self.settings['cameras'][0]}_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 24, (self.width, self.height))

        data = []
        frame_count = 0
        while True:
            frames = []
            show_frames = []
            frame_data = {}
            
            for i in range(len(self.cameras)):
                camera_name = self.settings['cameras'][i]
                ret, frame = self.cameras[i].read()
                frames.append(frame)
                if not save_preview:
                    shown_frame = frames[i].copy()
                else:
                    shown_frame = frame
                frame_data[camera_name] = capture_function(shown_frame, camera_name)
                show_frames.append(shown_frame)
                
            frame = self.frame_concatent(frames, self.reference_shape)
            show_frame = self.frame_concatent(show_frames, self.reference_shape)
            out.write(frame)
            cv2.imshow('Frames', show_frame)
            
            data.append(frame_data)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        json_data = []
        for frame_data in data:
            frame_json = {}
            for camera_name, positions in frame_data.items():
                frame_json[camera_name] = [pos.tolist() for pos in positions]
            json_data.append(frame_json)

        with open(f'output/{timestamp}.json', 'w') as f:
            json.dump(json_data, f, indent=2)
            
        out.release()
        cv2.destroyAllWindows()
        
    def frame_concatent(self, frames, reference_shape):
        for i in range(len(frames)):
            frames[i] = cv2.resize(frames[i], (reference_shape[1], reference_shape[0]))
        return np.concatenate(frames, axis=1)
    
    
class Localization(Capture):
    def __init__(self, cameras):
        super().__init__(cameras)
        self.cameras_mtx = {}
        self.cameras_dist = {}
        self.cameras_extrinsic = {}
        for camera_name in self.settings['cameras']:
            self.cameras_mtx[camera_name] = pickle.load(open(f'{self.settings["camera_parameter_path"]}/mtx_{camera_name}.pkl', 'rb'))
            self.cameras_dist[camera_name] = pickle.load(open(f'{self.settings["camera_parameter_path"]}/dist_{camera_name}.pkl', 'rb'))
            self.cameras_extrinsic[camera_name] = pickle.load(open(f'{self.settings["camera_parameter_path"]}/extrinsic_{camera_name}.pkl', 'rb'))
        
        self.detect_param_localization = cv2.aruco.DetectorParameters()
        self.detect_dict_localization = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.settings['aruco_dict_localization']))
        self.aruco_detector_localization = cv2.aruco.ArucoDetector(self.detect_dict_localization, self.detect_param_localization)
        
    def localization(self, frame, camera_name):
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, self.detect_dict_localization, parameters=self.detect_param_localization)
        frame_data = []
        
        for i in range(len(corners)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], self.settings['marker_size_localization'], self.cameras_mtx[camera_name], self.cameras_dist[camera_name])
            homogeneous_marker_point = np.eye(4)
            homogeneous_marker_point[:3, :3] = rvec
            homogeneous_marker_point[:3, 3] = tvec
            
            homogeneous_marker_point = self.cameras_extrinsic[camera_name] @ homogeneous_marker_point
            marker_info = (f"ID: {ids[i]} X: {homogeneous_marker_point[:3, 3][0]:.4f} Y: {homogeneous_marker_point[:3, 3][1]:.4f} Z: {homogeneous_marker_point[:3, 3][2]:.4f}")
            frame_data.append(homogeneous_marker_point[:3, 3])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            color = (0, 0, 255)
            cv2.putText(frame, marker_info, (int(corners[i][0][0][0]), int(corners[i][0][0][1])), font, font_scale, color, thickness, cv2.LINE_AA)
            
        return frame_data
    
        
def main():
    localization = Localization([cv2.VideoCapture(1), cv2.VideoCapture(3), cv2.VideoCapture(2)])
    localization.save_video(localization.localization, save_preview = True)
if __name__ == "__main__":
    main()
