import cv2
import numpy as np
import glob
import json
import os
import sys
import time
import random
import tqdm
import pickle
random.seed(time.time())

from settings_loader import settings

sys.path.append(os.path.dirname(os.path.abspath('.')))
from utils.frame_slicing import slicing_frame3_1, slicing_frame3_2

class ExtrinsicCalibrationCharuco:
    def __init__(self, settings):
        self.dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, settings.aruco_dict_external))
        self.board = cv2.aruco.CharucoBoard((settings.pattern_size_external[0], settings.pattern_size_external[1]), settings.pattern_square_size_external, settings.marker_size_external, self.dict)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dict, self.parameters)
        
        self.center_camera = settings.center_camera
        self.cameras = settings.cameras
        
        self.camera_points = []
        
        self.camera_mtx = {}
        self.camera_dist = {}
        self.camera_extrinsic = {}
        
        for camera_name in settings.cameras:
            self.camera_mtx[camera_name] = pickle.load(open(f'results/mtx_{camera_name}.pkl', 'rb'))
            self.camera_dist[camera_name] = pickle.load(open(f'results/dist_{camera_name}.pkl', 'rb'))
        
    def get_camera_points(self):
        image_path = f'../photos/multi_camera'
        images = glob.glob(f'{image_path}/*.jpg')
        
        for image_path in tqdm.tqdm(images):
            frame_rets = {}
            frame_points = {}
            image = cv2.imread(image_path)
            frames = slicing_frame3_1(image)
            for frame in frames:
                camera_name = self.cameras[0]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = self.detector.detectMarkers(gray)
                if ids is None:
                    frame_points[camera_name] = None
                    continue
                
                ret, corners, ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
                if not ret:
                    frame_points[camera_name] = None
                    continue
                
                ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(corners, ids, self.board, self.camera_mtx[camera_name], self.camera_dist[camera_name], None, None, useExtrinsicGuess=False)
                frame_points[camera_name] = (rvec, tvec)
                
            self.camera_points.append(frame_points)
            
    def calibrate(self, camera_name):
        camera_Rs = []
        camera_Ts = []
        
        center_Rs = []
        center_Ts = []
        
        for frame_points in self.camera_points:
            if frame_points[camera_name] is not None:
                camera_Rs.append(frame_points[camera_name][0])
                camera_Ts.append(frame_points[camera_name][1])
            if frame_points[self.center_camera] is not None:
                center_Rs.append(frame_points[self.center_camera][0])
                center_Ts.append(frame_points[self.center_camera][1])
        
        camera_Rs = np.array(camera_Rs)
        camera_Ts = np.array(camera_Ts)
        center_Rs = np.array(center_Rs)
        center_Ts = np.array(center_Ts)
        
        
        pass
    
    def evaluate(self):
        pass

### ----------------------------- Main function ----------------------------- ###
def main():
    extrinsic_calibration_charuco = ExtrinsicCalibrationCharuco(settings)
    extrinsic_calibration_charuco.get_camera_points()

if __name__ == '__main__':
    main()