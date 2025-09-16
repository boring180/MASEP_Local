import cv2
import numpy as np
import glob
import json
import os
import sys
import time
import random
import tqdm
random.seed(time.time())

from settings_loader import settings

sys.path.append(os.path.dirname(os.path.abspath('.')))
from utils.frame_slicing import slicing_frame3_1, slicing_frame3_2

def get_objp(number_of_internal_corners_x, number_of_internal_corners_y, square_size):
    objp = np.zeros((number_of_internal_corners_x * number_of_internal_corners_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:number_of_internal_corners_x,0:number_of_internal_corners_y].T.reshape(-1,2)
    objp = objp * square_size
    return objp

def get_points(settings, calibration_type):  
    if calibration_type == 'extrinsic' or settings.internal_callibration_type == 'multi':
        single_camera = False
    else:
        single_camera = True
    
    if single_camera:
        image_path = '../photos/single_camera'
    else:
        image_path = '../photos/multi_camera'
    
    images = glob.glob(f'{image_path}/*.jpg')
    shape = (0,0)
    imgpoints = []
    objpoints = []
    rets = []
    
    if calibration_type == 'intrinsic':
        pattern_size = settings.pattern_size_internal 
    else:
        pattern_size = settings.pattern_size_external
        
    pattern_size_X = pattern_size[0] - 1
    pattern_size_Y = pattern_size[1] - 1

    number_of_images = len(images)
    for fname in tqdm.tqdm(images):
        frame_imgpoints = np.zeros((len(settings.cameras), pattern_size_X * pattern_size_Y, 2))
        frame_objpoints = np.zeros((len(settings.cameras), pattern_size_X * pattern_size_Y, 3))
        frame_rets = (np.zeros(len(settings.cameras), dtype=bool) + True) & False
        
        if single_camera:
            camera_name = fname.split('/')[-1].split('.')[0].split('_')[0]
            if camera_name not in settings.cameras:
                continue
        
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if single_camera:
            frames = [gray]
        else:
            frames = slicing_frame3_1(gray)
        
        for i in range(len(frames)):
            frame = frames[i]
            shape = frame.shape
            
            if single_camera:
                i = settings.cameras.index(camera_name)

            ret, corners, objp = get_points_single_frame(frame, settings, calibration_type)
            
            if ret == True:
                frame_objpoints[i, :, :] = objp
                frame_imgpoints[i, :, :] = corners[:, 0, :]
                frame_rets[i] = True
                
            if single_camera:
                break
                
        imgpoints.append(frame_imgpoints)
        objpoints.append(frame_objpoints)
        rets.append(frame_rets)
        
    imgpoints = np.array(imgpoints)
    objpoints = np.array(objpoints)
    rets = np.array(rets)
    
    os.makedirs('chessboard_points', exist_ok=True)
    np.save(f'chessboard_points/{calibration_type}_rets.npy', rets)
    np.save(f'chessboard_points/{calibration_type}_object_points.npy', objpoints)
    np.save(f'chessboard_points/{calibration_type}_image_points.npy', imgpoints)
    np.save(f'chessboard_points/{calibration_type}_shape.npy', shape)
        
def get_points_single_frame(frame, settings, calibration_type):
    if calibration_type == 'intrinsic':
        pattern_size = settings.pattern_size_internal
        pattern_square_size = settings.pattern_square_size_internal
    elif calibration_type == 'extrinsic':
        pattern_size = settings.pattern_size_external
        pattern_square_size = settings.pattern_square_size_external
    
    internal_corners_X = pattern_size[0] - 1
    internal_corners_Y = pattern_size[1] - 1

    if settings.pattern_internal == 'chessboard':
        objp = get_objp(internal_corners_X, internal_corners_Y, pattern_square_size)
        ret, corners = cv2.findChessboardCorners(frame, (internal_corners_X,internal_corners_Y), flags=None)
    elif settings.pattern_internal == 'charuco':
        pass
    else:
        raise ValueError(f'Invalid pattern type: {settings.pattern_internal}')
    return ret, corners, objp