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

number_of_squares_x = 36
number_of_squares_y = 14
number_of_internal_corners_x = number_of_squares_x - 1
number_of_internal_corners_y = number_of_squares_y - 1
SQUARE_SIZE = 5.4/6.0  # in meters
cameras = ['cam2', 'cam3', 'wide', 'cam0', 'cam1']

image_path = '../photos/single_camera'

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(10,9,0)
objp = np.zeros((number_of_internal_corners_x * number_of_internal_corners_y,3), np.float32)
objp[:,:2] = np.mgrid[0:number_of_internal_corners_x,0:number_of_internal_corners_y].T.reshape(-1,2)
objp = objp * SQUARE_SIZE

        
def intrinsic_calibration(camera_name):
    object_points_files = os.listdir('chessboard_points/')
    if f'{camera_name}_object_points.json' not in object_points_files or f'{camera_name}_image_points.json' not in object_points_files:
        get_points(image_path, number_of_internal_corners_x, number_of_internal_corners_y, SQUARE_SIZE, camera_name)
    
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

    
def main():
    os.makedirs('results', exist_ok=True)
    results = os.listdir('results')
    for camera_name in cameras:
        if f'intrinsic_{camera_name}.json' in results:
            print(f'Intrinsic calibration for {camera_name} already done')
        else:
            print(f'Intrinsic calibration for {camera_name} not done, start to do intrinsic calibration')
            intrinsic_calibration(camera_name)
    
if __name__ == '__main__':
    main()