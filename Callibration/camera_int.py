import cv2
import numpy as np
import json
import os
import sys
import time
import random
random.seed(time.time())

from get_points import get_points

sys.path.append(os.path.dirname(os.path.abspath('.')))
from utils.frame_concatent import resize_with_padding

cameras = ['cam2', 'cam3', 'wide', 'cam0', 'cam1']

image_path = '../photos/single_camera'

        
def single_camera_intrinsic_calibration(camera_name):
    object_points_files = os.listdir('chessboard_points/')
    if f'{camera_name}_object_points.json' not in object_points_files or f'{camera_name}_image_points.json' not in object_points_files:
        get_points(image_path, camera_name)
    
    with open(f'chessboard_points/{camera_name}_object_points.json', 'r') as f:
        obj_pts = np.array(json.load(f))
    with open(f'chessboard_points/{camera_name}_image_points.json', 'r') as f:
        img_pts = np.array(json.load(f))
    with open(f'chessboard_points/shape.json', 'r') as f:
        shape = json.load(f)
        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts.astype(np.float32), img_pts.astype(np.float32), shape[::-1], None, None)
    
    with open(f'results/intrinsic_{camera_name}.json', 'w') as f:
        json.dump({'mtx': mtx.tolist(), 'dist': dist.tolist()}, f)

    mean_error = 0
    for i in range(len(obj_pts)):
        imgpoints2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
        # Reshape imgpoints2 to match img_pts dimensions
        imgpoints2 = imgpoints2.reshape(-1, 2)
        error = cv2.norm(img_pts[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error = mean_error / len(obj_pts)
    print(f'{camera_name} has mean error: {mean_error}')
            
    return

def intrinsic_calibration():
    if os.path.exists('results/rets.json') and os.path.exists('results/object_points.json') and os.path.exists('results/image_points.json') and os.path.exists('results/shape.json'):
        print('Image points already exist')
    else:
        print('Image points not exist, start to get image points')
        get_points()
    
    with open('results/rets.json', 'r') as f:
        rets = np.array(json.load(f))
    with open('results/object_points.json', 'r') as f:
        objpoints = np.array(json.load(f))
    with open('results/image_points.json', 'r') as f:
        imgpoints = np.array(json.load(f))
    with open('results/shape.json', 'r') as f:
        shape = tuple(json.load(f))
    
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
        
        with open(f'results/intrinsic_{camera_name}.json', 'w') as f:
            json.dump({'mtx': mtx.tolist(), 'dist': dist.tolist()}, f)
        
        print(f'{camera_name} has {number_of_success_images}/{number_of_images} successful images')
        
        mean_error = 0
        for i in range(number_of_success_images):
            imgpoints2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
            # Reshape imgpoints2 to match img_pts dimensions
            imgpoints2 = imgpoints2.reshape(-1, 2)
            error = cv2.norm(img_pts[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        mean_error = mean_error / number_of_success_images
        print(f'{camera_name} has mean error: {mean_error}')
            
    return

    
def main():
    os.makedirs('results', exist_ok=True)
    results = os.listdir('results')
    for camera_name in cameras:
        if f'intrinsic_{camera_name}.json' in results:
            print(f'Intrinsic calibration for {camera_name} already done')
        else:
            print(f'Intrinsic calibration for {camera_name} not done, start to do intrinsic calibration')
            single_camera_intrinsic_calibration(camera_name)
    
if __name__ == '__main__':
    main()