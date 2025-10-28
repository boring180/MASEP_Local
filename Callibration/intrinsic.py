import cv2
import numpy as np
import os
import sys
import pickle
import glob
import json
import tqdm

class IntrinsicCalibration:
    def __init__(self):
        try:
            with open('setting.json', 'r') as f:
                self.settings = json.load(f)
        except:
            raise Exception(f"Error loading settings from setting.json")
        
        self._internal_corners_X = self.settings['pattern_size_internal'][0] - 1
        self._internal_corners_Y = self.settings['pattern_size_internal'][1] - 1
        self._objp = np.zeros((self._internal_corners_X * self._internal_corners_Y,3), np.float32)
        self._objp[:,:2] = np.mgrid[0:self._internal_corners_X,0:self._internal_corners_Y].T.reshape(-1,2)
        self._objp = self._objp * self.settings['pattern_square_size_internal']
        
        self.camera_object_points = {camera_name: [] for camera_name in self.settings['cameras']}
        self.camera_image_points = {camera_name: [] for camera_name in self.settings['cameras']}
        self.shape = None
    
    ## ----------------------------- Get camera points ----------------------------- ##
    def get_points(self):
        if self.settings['internal_callibration_type'] == 'single':
            single_camera = True
        else:
            single_camera = False
            
        if single_camera:
            image_path = '../photos/single_camera'
        else:
            image_path = '../photos/multi_camera'
            
        images = glob.glob(f'{image_path}/*.jpg')
        for image_path in tqdm.tqdm(images):            
            if single_camera:
                camera_name = image_path.split('/')[-1].split('.')[0].split('_')[0]
                if camera_name not in self.settings['cameras']:
                    continue
        
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if single_camera:
                frames = [gray]
            else:
                frames = self.slicing_frame3_1(gray)
            
            for i in range(len(frames)):
                frame = frames[i]
                self.shape = frame.shape
                
                if not single_camera:
                    camera_name = self.settings['cameras'][i]

                ret, imgp = cv2.findChessboardCorners(frame, (self._internal_corners_X, self._internal_corners_Y), flags=None)
                
                if ret == True:
                    self.camera_object_points[camera_name].append(self._objp)
                    self.camera_image_points[camera_name].append(imgp[:, 0, :])
                
    def save_points(self):
        pickle.dump(self.camera_object_points, open(f'results/intrinsic_camera_object_points.pkl', 'wb'))
        pickle.dump(self.camera_image_points, open(f'results/intrinsic_camera_image_points.pkl', 'wb'))
        pickle.dump(self.shape, open(f'results/intrinsic_shape.pkl', 'wb'))
        
    def load_points(self):
        self.camera_object_points = pickle.load(open(f'results/intrinsic_camera_object_points.pkl', 'rb'))
        self.camera_image_points = pickle.load(open(f'results/intrinsic_camera_image_points.pkl', 'rb'))
        self.shape = pickle.load(open(f'results/intrinsic_shape.pkl', 'rb'))
    
    ## ----------------------------- Calibrate intrinsic ----------------------------- ##
    def calibrate_intrinsic(self):
        for camera_name in self.settings['cameras']:
            imgpoints = self.camera_image_points[camera_name]
            objpoints = self.camera_object_points[camera_name]
            shape = self.shape
            
            termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            flags = None
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape[::-1], None, None, flags=flags, criteria=termination_criteria)
            pickle.dump(mtx, open(f'results/mtx_{camera_name}.pkl', 'wb'))
            with open(f'results/intrinsic_calibration.log', 'a') as f:
                f.write(f'{camera_name} has mtx: {mtx}\n')
            pickle.dump(dist, open(f'results/dist_{camera_name}.pkl', 'wb'))
            with open(f'results/intrinsic_calibration.log', 'a') as f:
                f.write(f'{camera_name} has dist: {dist}\n')
            
            mean_error = 0
            num_points = 0
            for i in range(len(imgpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2[:, 0, :], cv2.NORM_L2)
                num_points += len(imgpoints[i])
                mean_error += error
                
            mean_error /= num_points
            
            print(f'{camera_name} has reprojection error: {mean_error}')
            with open(f'results/intrinsic_calibration.log', 'a') as f:
                f.write(f'{camera_name} has reprojection error: {mean_error}\n')
    
    ## ----------------------------- Evaluate ----------------------------- ##
    def evaluate(self):
        pass

    def visualize(self):
        pass

    ## ----------------------------- Helpers ----------------------------- ##
    def slicing_frame3_1(self, frame):
        width = frame.shape[1] // 3
        height = frame.shape[0]
        frames = [frame[:, 0:width], frame[:, width:width*2], frame[:, width*2:width*3]]
        return frames

def main():
    os.makedirs('results', exist_ok=True)
    
    intrinsic = IntrinsicCalibration()
    # intrinsic.get_points()
    # intrinsic.save_points()
    intrinsic.load_points()
    intrinsic.calibrate_intrinsic()
    
if __name__ == '__main__':
    main() 