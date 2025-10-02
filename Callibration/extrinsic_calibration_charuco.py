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
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
random.seed(time.time())

from settings_loader import settings

sys.path.append(os.path.dirname(os.path.abspath('.')))
from utils.frame_slicing import slicing_frame3_1, slicing_frame3_2

class ExtrinsicCalibrationCharuco:
    def __init__(self, settings):
        self.dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, settings.aruco_dict_external))
        self.board = cv2.aruco.CharucoBoard((settings.pattern_size_external[1], settings.pattern_size_external[0]), settings.pattern_square_size_external, settings.marker_size_external, self.dict)
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
            for i in range(len(frames)):
                camera_name = self.cameras[i]
                gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = self.detector.detectMarkers(gray)
                if ids is None:
                    frame_points[camera_name] = None
                    continue
                
                ret, corners, ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
                if ret:
                    number_of_corners = len(corners)
                if not ret:
                    number_of_corners = 0

                if ret:
                    name = f"{image_path.split('/')[-1].split('.')[0]}_{camera_name}"
                    self.draw_points(gray, corners, ids, name)
                
                if number_of_corners < 6:
                    frame_points[camera_name] = None
                    continue
                
                ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(corners, ids, self.board, self.camera_mtx[camera_name], self.camera_dist[camera_name], None, None, useExtrinsicGuess=False)
                
                frame_points[camera_name] = (rvec, tvec, number_of_corners)
                
            self.camera_points.append(frame_points)
            
    def calibrate(self, camera_name, weighted = False):
        if camera_name == self.center_camera:
            self.camera_extrinsic[camera_name] = np.eye(4)
            pickle.dump(self.camera_extrinsic[camera_name], open(f'results/extrinsic_{camera_name}.pkl', 'wb'))
            return
        
        
        camera_Rs = []
        camera_Ts = []
        camera_number_of_corners = []
        
        center_Rs = []
        center_Ts = []
        center_number_of_corners = []
        
        
        for frame_points in self.camera_points:
            if frame_points[camera_name] is not None and frame_points[self.center_camera] is not None:
                camera_Rs.append(frame_points[camera_name][0][:, 0])
                camera_Ts.append(frame_points[camera_name][1][:, 0])
                camera_number_of_corners.append(frame_points[camera_name][2])
                center_Rs.append(frame_points[self.center_camera][0][:, 0])
                center_Ts.append(frame_points[self.center_camera][1][:, 0])
                center_number_of_corners.append(frame_points[self.center_camera][2])
        
        camera_Rs = np.array(camera_Rs)
        camera_Ts = np.array(camera_Ts)
        center_Rs = np.array(center_Rs)
        center_Ts = np.array(center_Ts)
        
        camera_number_of_corners = np.array(camera_number_of_corners)
        center_number_of_corners = np.array(center_number_of_corners)
        
        if weighted:
            weight = camera_number_of_corners + center_number_of_corners
        else:
            weight = np.ones(camera_Rs.shape[0])
        
        print(f'{camera_name} has {camera_Ts.shape[0]} points')
        with open(f'results/extrinsic_calibration.log', 'a') as f:
            f.write(f'{camera_name} has {camera_Ts.shape[0]} points\n')
        
        Rotation, transformation, rmsd = self.point_cloud_matching(camera_Ts, center_Ts, weight)
        print(f'{camera_name} has RMSD: {rmsd}')
        with open(f'results/extrinsic_calibration.log', 'a') as f:
            f.write(f'{camera_name} has RMSD: {rmsd}\n')
        
        self.camera_extrinsic[camera_name] = np.eye(4)
        self.camera_extrinsic[camera_name][:3, :3] = Rotation.as_matrix()
        self.camera_extrinsic[camera_name][:3, 3] = transformation
        
        print(f'{camera_name} transformation matrix: {self.camera_extrinsic[camera_name]}')
        with open(f'results/extrinsic_calibration.log', 'a') as f:
            f.write(f'{camera_name} transformation matrix: {self.camera_extrinsic[camera_name]}\n')
        
        pickle.dump(self.camera_extrinsic[camera_name], open(f'results/extrinsic_{camera_name}.pkl', 'wb'))
    
    def ransac(self):
        pass
    
    # P, Q: (N, 3)
    # weight: (N,)
    def point_cloud_matching(self, P, Q, weight):
        assert P.shape == Q.shape
        assert weight.shape[0] == P.shape[0]

        transformation = (Q - P) * weight[:, np.newaxis]
        transformation = transformation.sum(axis=0) / weight.sum()
        
        P_prime = P + transformation
        
        Rotation, rmsd = R.align_vectors(Q, P_prime, weights=weight)
        return Rotation, transformation, rmsd
    
    def evaluate(self):
        fig = plt.figure(figsize=(15, 10))
        self.visualize_camera_location(settings, fig, 45, 0, 0, i=1)
        self.visualize_camera_location(settings, fig, elev=45, azim=45, roll=0, i=2)
        self.visualize_camera_location(settings, fig, elev=-90, azim=90, roll=0, i=3)
        plt.savefig('results/visualize_ext.png')
        
        
    def visualize_camera_location(self, settings, fig, elev, azim, roll, i):
        ax = fig.add_subplot(1, 3, i, projection='3d', elev=elev, azim=azim, roll=roll)
        ax.set_box_aspect([1,1,1])
        length = 0.5
        width = 0.5
        height = 0.5
        
        x = [-length/2, length/2, length/2, -length/2]
        y = [-width/2, -width/2, width/2, width/2]
        for i in range(4):
            ax.plot([x[i], x[i]], [y[i], y[i]], [0, height], 'b-')
            
        x = [-length/2, length/2, length/2, -length/2]
        z = [0, 0, height, height]
        for i in range(4):
            ax.plot([x[i], x[i]], [-width/2, width/2], [z[i], z[i]], 'b-')
            
        y = [-length/2, length/2, length/2, -length/2]
        z = [0, 0, height, height]
        for i in range(4):
            ax.plot([-length/2, length/2], [y[i], y[i]], [z[i], z[i]], 'b-')
        
        for camera_name in settings.cameras:
            colors = ['red', 'green', 'blue']
            axis = np.eye(3) * settings.pattern_square_size_external
            for i in range(3):
                orientation = self.camera_extrinsic[camera_name][:3, :3] @ axis[i, :]
                x = self.camera_extrinsic[camera_name][0, 3]
                y = self.camera_extrinsic[camera_name][1, 3]
                z = self.camera_extrinsic[camera_name][2, 3]
                u = orientation[0]
                v = orientation[1]
                w = orientation[2]
                ax.quiver(x, y, z, u, v, w, color = colors[i], length=1)
                
                if i == 0:
                    ax.text(x, y, z, camera_name, color = 'black')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    
    def draw_points(self, gray, corners, ids, name):
        for i in range(len(corners)):
            cv2.circle(gray, (int(corners[i][0][0]), int(corners[i][0][1])), 5, (0, 0, 255), -1)
        os.makedirs('debug_frames', exist_ok=True)
        cv2.imwrite(f'debug_frames/{name}.jpg', gray)
        

### ----------------------------- Main function ----------------------------- ###
def main():
    extrinsic_calibration_charuco = ExtrinsicCalibrationCharuco(settings)
    extrinsic_calibration_charuco.get_camera_points()
    extrinsic_calibration_charuco.calibrate('cam0', weighted=True)
    extrinsic_calibration_charuco.calibrate('cam1', weighted=True)
    extrinsic_calibration_charuco.calibrate('cam2', weighted=True)
    extrinsic_calibration_charuco.evaluate()

if __name__ == '__main__':
    main()