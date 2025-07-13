import cv2
import numpy as np
import json
import os
import sys
import time
import random
random.seed(time.time())

sys.path.append(os.path.dirname(os.path.abspath('.')))
import utils.frame_concatent as resize_with_padding

number_of_squares_x = 11
number_of_internal_corners_x = number_of_squares_x - 1
number_of_squares_y = 8
number_of_internal_corners_y = number_of_squares_y - 1
SQUARE_SIZE = 0.023 # in meters
CAMERA_NAME = 'cam0'

image_path = '../photos/single_camera'

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(10,9,0)
objp = np.zeros((number_of_internal_corners_x * number_of_internal_corners_y,3), np.float32)
objp[:,:2] = np.mgrid[0:number_of_internal_corners_x,0:number_of_internal_corners_y].T.reshape(-1,2)
objp = objp * SQUARE_SIZE

        
def intrinsic_calibration():
    images = os.listdir(image_path)
    obj_pts = []
    img_pts = []
    
    for image in images:
        if image.split('.')[0].split('_')[-1] != CAMERA_NAME:
            continue
        
        img = cv2.imread(f'{image_path}/{image}')
        img = resize_with_padding(img, img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape
        ret, corners = cv2.findChessboardCorners(gray, (number_of_internal_corners_x,number_of_internal_corners_y), None)
        
        # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # img = cv2.rotate(img, cv2.ROTATE_180)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria=criteria)
            obj_pts.append(objp)
            img_pts.append(corners2[:, 0, :])
                
            cv2.drawChessboardCorners(img, (number_of_internal_corners_x,number_of_internal_corners_y), corners2, ret)
            cv2.imwrite(f'chessboard_points/{image}', img)
    
            
        obj_pts = np.array(obj_pts)
        img_pts = np.array(img_pts)
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, shape[::-1], None, None)
        
        with open(f'results/intrinsic_{CAMERA_NAME}.json', 'w') as f:
            json.dump({'mtx': mtx.tolist(), 'dist': dist.tolist()}, f)
        
        print(f'{CAMERA_NAME} has {len(obj_pts)}/{len(images)} successful images')

        mean_error = 0
        for i in range(len(obj_pts)):
            imgpoints2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
            # Reshape imgpoints2 to match img_pts dimensions
            imgpoints2 = imgpoints2.reshape(-1, 2)
            error = cv2.norm(img_pts[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        mean_error = mean_error / len(obj_pts)
        print(f'{CAMERA_NAME} has mean error: {mean_error}')
            
    return

    
def main():
            
    results = os.listdir('results')
    if f'intrinsic_{CAMERA_NAME}.json' in results:
        print(f'Intrinsic calibration for {CAMERA_NAME} already done')
        pass
    else:
        print(f'Intrinsic calibration for {CAMERA_NAME} not done, start to do intrinsic calibration')
        intrinsic_calibration()
    
if __name__ == '__main__':
    main()