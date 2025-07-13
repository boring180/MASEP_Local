import cv2
import numpy as np
import glob
import json
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import sys
import time
import random
random.seed(time.time())

sys.path.append(os.path.dirname(os.path.abspath('.')))
import utils.frame_slicing as frame_slicing
import utils.frame_concatent as frame_concatent

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


def get_points():
    imgpoints = [] # 2D points in image plane
    objpoints = [] # 3D points in world coordinate system
    rets = [] # boolean values
    
    images = glob.glob(f'{image_path}/*.jpg')
    
    os.makedirs('chessboard_points', exist_ok=True)
    
    shape = (0,0)
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        frames = frame_slicing.slicing_frame(gray)[:5]
        
        frame_imgpoints = np.zeros((len(frames), number_of_internal_corners_x * number_of_internal_corners_y, 2))
        frame_objpoints = np.zeros((len(frames), number_of_internal_corners_x * number_of_internal_corners_y, 3))
        frame_rets = np.zeros(len(frames), dtype=bool) + True
        
        for i in range(len(frames)):
            frame = frames[i]
            shape = frame.shape
            
            ret, corners = cv2.findChessboardCorners(frame, (number_of_internal_corners_x,number_of_internal_corners_y), None)
            
            if ret == True:
                corners2 = cv2.cornerSubPix(frame, corners, (11,11), (-1,-1), criteria=criteria)
                frame_objpoints[i, :, :] = objp
                frame_imgpoints[i, :, :] = corners2[:, 0, :]
                
                cv2.drawChessboardCorners(frame, (number_of_internal_corners_x,number_of_internal_corners_y), corners2, ret)
                
            else:
                frame_rets[i] = False
                
        imgpoints.append(frame_imgpoints)
        objpoints.append(frame_objpoints)
        rets.append(frame_rets)
        
        frame_with_corners = frame_concatent.concatent_frame(frames)
        cv2.imwrite(f'chessboard_points/{fname.split("/")[-1]}', frame_with_corners)
        
    imgpoints = np.array(imgpoints)
    objpoints = np.array(objpoints)
    rets = np.array(rets)
    
    os.makedirs('results', exist_ok=True)
        
    if rets.sum() == 0:
        print('No chessboard found in any image')
        return
    
    with open('results/rets.json', 'w') as f:
        json.dump(rets.tolist(), f)
    with open('results/object_points.json', 'w') as f:
        json.dump(objpoints.tolist(), f)
    with open('results/image_points.json', 'w') as f:
        json.dump(imgpoints.tolist(), f)
    with open('results/shape.json', 'w') as f:
        json.dump(shape, f)