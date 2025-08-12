import cv2
import numpy as np
import json
import os
import sys
import time
import random
random.seed(time.time())

from get_points import get_points

from settings_loader import settings

sys.path.append(os.path.dirname(os.path.abspath('.')))
from utils.frame_concatent import resize_with_padding

cameras = settings.cameras
        
def single_camera_intrinsic_calibration(camera_name):
    object_points_files = os.listdir('chessboard_points/')
    if f'{camera_name}_object_points.npy' not in object_points_files or f'{camera_name}_image_points.npy' not in object_points_files:
        get_points(settings, calibration_type='single')
    
    obj_pts = np.load(f'chessboard_points/{camera_name}_object_points.npy')
    img_pts = np.load(f'chessboard_points/{camera_name}_image_points.npy')
    shape = np.load(f'chessboard_points/shape.npy')
        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts.astype(np.float32), img_pts.astype(np.float32), shape[::-1], None, None)
    
    np.save(f'results/intrinsic_{camera_name}.npy', np.array([mtx, dist]))

    mean_error = 0
    for i in range(len(obj_pts)):
        imgpoints_reprojected, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
        imgpoints_reprojected = imgpoints_reprojected.reshape(-1, 2)
        error = cv2.norm(img_pts[i], imgpoints_reprojected, cv2.NORM_L2) / len(imgpoints_reprojected)
        mean_error += error
    mean_error = mean_error / len(obj_pts)
    print(f'{camera_name} has reprojection error: {mean_error}')
    return

def multi_camera_intrinsic_calibration():
    if os.path.exists('results/rets.npy') and os.path.exists('results/object_points.npy') and os.path.exists('results/image_points.npy') and os.path.exists('results/shape.npy'):
        print('Image points already exist')
    else:
        print('Image points not exist, start to get image points')
        get_points(settings, calibration_type='multi')
    
    rets = np.load('results/rets.npy')
    objpoints = np.load('results/object_points.npy')
    imgpoints = np.load('results/image_points.npy')
    shape = np.load('results/shape.npy')
    
    for camera_name in cameras:
        imgpoints_camera = imgpoints[:, cameras.index(camera_name), :, :]
        objpoints_camera = objpoints[:, cameras.index(camera_name), :, :]
        rets_camera = rets[:, cameras.index(camera_name)]
        imgpoints_camera = imgpoints_camera[rets_camera, :, :]
        objpoints_camera = objpoints_camera[rets_camera, :, :]
        
        number_of_images = len(rets_camera)
        number_of_success_images = np.sum(rets_camera.astype(int))
        
        obj_pts = []
        img_pts = []
        for i in range(number_of_success_images):
            obj_pts.append(objpoints_camera[i, :, :].astype(np.float32))
            img_pts.append(imgpoints_camera[i, :, :].astype(np.float32))
            
        obj_pts = np.array(obj_pts)
        img_pts = np.array(img_pts)        
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, shape[::-1], None, None)
        
        np.save(f'results/intrinsic_{camera_name}.npy', np.array([mtx, dist]))
        
        print(f'{camera_name} has {number_of_success_images}/{number_of_images} successful images')
        
        mean_error = 0
        for i in range(number_of_success_images):
            imgpoints_reprojected, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
            imgpoints_reprojected = imgpoints_reprojected.reshape(-1, 2)
            error = cv2.norm(img_pts[i], imgpoints_reprojected, cv2.NORM_L2) / len(imgpoints_reprojected)
            mean_error += error
        mean_error = mean_error / number_of_success_images
        print(f'{camera_name} has reprojection error: {mean_error}')
            
    return

    
def main():
    os.makedirs('results', exist_ok=True)
    results = os.listdir('results')
    for camera_name in cameras:
        if f'intrinsic_{camera_name}.npy' in results:
            print(f'Intrinsic calibration for {camera_name} already done')
        else:
            print(f'Intrinsic calibration for {camera_name} not done, start to do intrinsic calibration')
            if settings.internal_callibration_type == 'single':
                single_camera_intrinsic_calibration(camera_name)
            elif settings.internal_callibration_type == 'multi':
                multi_camera_intrinsic_calibration()
            else:
                raise ValueError(f'Invalid internal callibration type: {settings.internal_callibration_type}')
    
if __name__ == '__main__':
    main()