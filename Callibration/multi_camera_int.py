import cv2
import numpy as np
import json
import os
import time
import random
random.seed(time.time())

from get_points import get_points

number_of_squares_x = 11
number_of_internal_corners_x = number_of_squares_x - 1
number_of_squares_y = 8
number_of_internal_corners_y = number_of_squares_y - 1
SQUARE_SIZE = 0.023 # in meters
cameras = ['cam2', 'cam3', 'wide', 'cam0', 'cam1']

image_path = '../photos/multi_camera'

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(10,9,0)
objp = np.zeros((number_of_internal_corners_x * number_of_internal_corners_y,3), np.float32)
objp[:,:2] = np.mgrid[0:number_of_internal_corners_x,0:number_of_internal_corners_y].T.reshape(-1,2)
objp = objp * SQUARE_SIZE

        
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
            
    results = os.listdir('results')
    if 'intrinsic_cam0.json' in results and 'intrinsic_cam1.json' in results and 'intrinsic_cam2.json' in results and 'intrinsic_cam3.json' in results and 'intrinsic_wide.json' in results:
        print('Intrinsic calibration already done')
        pass
    else:
        print('Intrinsic calibration not done, start to do intrinsic calibration')
        intrinsic_calibration()
    
if __name__ == '__main__':
    main()