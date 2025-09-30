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
random.seed(time.time())

from settings_loader import settings

sys.path.append(os.path.dirname(os.path.abspath('.')))
from utils.frame_slicing import slicing_frame3_1, slicing_frame3_2

### ----------------------------- Charuco pattern settings ----------------------------- ###
def get_charuco_settings(settings, calibration_type):
    if calibration_type == 'intrinsic':
        pattern_size = settings.pattern_size_internal
        pattern_square_size = settings.pattern_square_size_internal
        marker_size = settings.marker_size_internal
        aruco_dict = settings.aruco_dict_internal
    elif calibration_type == 'extrinsic':
        pattern_size = settings.pattern_size_external
        pattern_square_size = settings.pattern_square_size_external
        marker_size = settings.marker_size_external
        aruco_dict = settings.aruco_dict_external
    SQUARE_SIZE = pattern_square_size
    MARKER_SIZE = marker_size
    DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict))
    board = cv2.aruco.CharucoBoard((pattern_size[0], pattern_size[1]), SQUARE_SIZE, MARKER_SIZE, DICT)
    parameters = cv2.aruco.DetectorParameters()
    return board, parameters, DICT

### ----------------------------- Chessboard pattern settings ----------------------------- ###
def get_objp(number_of_internal_corners_x, number_of_internal_corners_y, square_size):
    objp = np.zeros((number_of_internal_corners_x * number_of_internal_corners_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:number_of_internal_corners_x,0:number_of_internal_corners_y].T.reshape(-1,2)
    objp = objp * square_size
    return objp


### ----------------------------- Get points from images----------------------------- ###
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
        frame_imgpoints = {}
        frame_objpoints = {}
        frame_rets = {camera_name: False for camera_name in settings.cameras}
        
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
            
            if not single_camera:
                camera_name = settings.cameras[i]

            ret, imgp, objp = get_points_single_frame(frame, settings, calibration_type)
            
            if ret == True:
                frame_objpoints.update({camera_name: objp})
                frame_imgpoints.update({camera_name: imgp[:, 0, :]})
                frame_rets.update({camera_name: True})
            else:
                frame_rets.update({camera_name: False})
                
        imgpoints.append(frame_imgpoints)
        objpoints.append(frame_objpoints)
        rets.append(frame_rets)
    
    os.makedirs('chessboard_points', exist_ok=True)
    pickle.dump(rets, open(f'chessboard_points/{calibration_type}_rets.pkl', 'wb'))
    pickle.dump(objpoints, open(f'chessboard_points/{calibration_type}_object_points.pkl', 'wb'))
    pickle.dump(imgpoints, open(f'chessboard_points/{calibration_type}_image_points.pkl', 'wb'))
    pickle.dump(shape, open(f'chessboard_points/{calibration_type}_shape.pkl', 'wb'))
        
### ----------------------------- Get points from single frame----------------------------- ###
def get_points_single_frame(frame, settings, calibration_type):
    ## ----------------------------- Chessboard pattern ----------------------------- ##
    if settings.pattern_internal == 'chessboard':
        if calibration_type == 'intrinsic':
            pattern_size = settings.pattern_size_internal
            pattern_square_size = settings.pattern_square_size_internal
        elif calibration_type == 'extrinsic':
            pattern_size = settings.pattern_size_external
            pattern_square_size = settings.pattern_square_size_external
        internal_corners_X = pattern_size[0] - 1
        internal_corners_Y = pattern_size[1] - 1
        
        objp = get_objp(internal_corners_X, internal_corners_Y, pattern_square_size)
        ret, corners = cv2.findChessboardCorners(frame, (internal_corners_X,internal_corners_Y), flags=None)

        return ret, corners, objp
    
    ## ----------------------------- Charuco pattern ----------------------------- ##
    elif settings.pattern_internal == 'charuco':
        board, parameters, aruco_dict = get_charuco_settings(settings, calibration_type)
        charuco_corners, charuco_ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        ret = charuco_ids is not None
        if ret:
            objp, imgp = board.matchImagePoints(charuco_corners, charuco_ids)
        else:
            objp = None
            imgp = None
        return ret, imgp, objp
    
    ## ----------------------------- Invalid pattern type ----------------------------- ##
    else:
        raise ValueError(f'Invalid pattern type: {settings.pattern_internal}')

### ----------------------------- Main function ----------------------------- ###
def main():
    args = sys.argv[1:]
    if 'intrinsic' in args:
        get_points(settings, calibration_type='intrinsic')
    if 'extrinsic' in args:
        get_points(settings, calibration_type='extrinsic')

if __name__ == '__main__':
    main()