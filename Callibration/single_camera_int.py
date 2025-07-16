import cv2
import numpy as np
import json
import os
import sys
import time
import random
random.seed(time.time())

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
    images = os.listdir(image_path)
    obj_pts = []
    img_pts = []
    image_count = 0

    for image in images:
        if image.split('.')[0].split('_')[0] != camera_name:
            continue        
        
        image_count += 1
        
        img = cv2.imread(f'{image_path}/{image}')
        img = resize_with_padding(img, np.max(img.shape), np.max(img.shape))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, (number_of_internal_corners_x,number_of_internal_corners_y), flags=flags)
        
        if camera_name == 'cam2':
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif camera_name == 'cam3':
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif camera_name == 'cam0':
            img = cv2.rotate(img, cv2.ROTATE_180)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria=criteria)
            obj_pts.append(objp)
            img_pts.append(corners2[:, 0, :])
                
            cv2.drawChessboardCorners(img, (number_of_internal_corners_x,number_of_internal_corners_y), corners2, ret)
            cv2.imwrite(f'chessboard_points/{image}', img)
    
            
    obj_pts = np.array(obj_pts)
    img_pts = np.array(img_pts)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, shape[::-1], None, None)
    
    with open(f'results/intrinsic_{camera_name}.json', 'w') as f:
        json.dump({'mtx': mtx.tolist(), 'dist': dist.tolist()}, f)
    
    print(f'{camera_name} has {len(obj_pts)}/{image_count} successful images')

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
            
    results = os.listdir('results')
    for camera_name in cameras:
        if f'intrinsic_{camera_name}.json' in results:
            print(f'Intrinsic calibration for {camera_name} already done')
        else:
            print(f'Intrinsic calibration for {camera_name} not done, start to do intrinsic calibration')
            intrinsic_calibration(camera_name)
    
if __name__ == '__main__':
    main()