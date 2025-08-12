import cv2
import numpy as np
import glob
import json
import os
import sys
import time
import random
random.seed(time.time())

from settings_loader import settings

sys.path.append(os.path.dirname(os.path.abspath('.')))
from utils.frame_slicing import slicing_frame
from utils.frame_concatent import resize_with_padding

def get_objp(number_of_internal_corners_x, number_of_internal_corners_y, square_size):
    objp = np.zeros((number_of_internal_corners_x * number_of_internal_corners_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:number_of_internal_corners_x,0:number_of_internal_corners_y].T.reshape(-1,2)
    objp = objp * square_size
    return objp

def get_points(settings, calibration_type):            
    
    images = glob.glob(f'{image_path}/*.jpg')
    shape = (0,0)
    imgpoints = []
    objpoints = []
    rets = []
    
    
    for fname in images:
        if camera_name is not None and camera_name not in fname:
            continue
        
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if camera_name is None:
            frames = slicing_frame(gray)
        else:
            gray = resize_with_padding(gray, np.max(gray.shape), np.max(gray.shape))
            if camera_name == 'cam2':
                gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif camera_name == 'cam3':
                gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
            elif camera_name == 'cam0':
                gray = cv2.rotate(gray, cv2.ROTATE_180)
            frames = [gray]
        
        frame_imgpoints = np.zeros((len(frames), number_of_internal_corners_x * number_of_internal_corners_y, 2))
        frame_objpoints = np.zeros((len(frames), number_of_internal_corners_x * number_of_internal_corners_y, 3))
        frame_rets = np.zeros(len(frames), dtype=bool) + True
        
        for i in range(len(frames)):
            frame = frames[i]
            shape = frame.shape

            ret, corners, objp = get_points_single_frame(frame)
            
            if ret == True:
                frame_objpoints[i, :, :] = objp
                frame_imgpoints[i, :, :] = corners[:, 0, :]
                
            else:
                frame_rets[i] = False
                
        imgpoints.append(frame_imgpoints)
        objpoints.append(frame_objpoints)
        rets.append(frame_rets)
        
    imgpoints = np.array(imgpoints)
    objpoints = np.array(objpoints)
    rets = np.array(rets)
    
    os.makedirs('chessboard_points', exist_ok=True)
    if camera_name is not None:
        rets = rets[:, 0]
        objpoints = objpoints[rets, 0, :, :]
        imgpoints = imgpoints[rets, 0, :, :]
        with open(f'chessboard_points/{camera_name}_object_points.json', 'w') as f:
            json.dump(objpoints.tolist(), f)
        with open(f'chessboard_points/{camera_name}_image_points.json', 'w') as f:
            json.dump(imgpoints.tolist(), f)
        print(f'{camera_name} has {rets.sum()}/{rets.shape[0]} successful images')
    else:
        with open('chessboard_points/rets.json', 'w') as f:
            json.dump(rets.tolist(), f)
        with open('chessboard_points/object_points.json', 'w') as f:
            json.dump(objpoints.tolist(), f)
        with open('chessboard_points/image_points.json', 'w') as f:
            json.dump(imgpoints.tolist(), f)

    with open('chessboard_points/shape.json', 'w') as f:
        json.dump(shape, f)
        
def get_points_single_frame(frame, number_of_internal_corners_x, number_of_internal_corners_y, square_size):
    objp = get_objp(number_of_internal_corners_x, number_of_internal_corners_y, square_size)
    ret, corners = cv2.findChessboardCorners(frame, (number_of_internal_corners_x,number_of_internal_corners_y), flags=None)
    return ret, corners, objp
        
if __name__ == '__main__':
    image_path = '../photos/single_camera'
    get_points(image_path, camera_name = 'cam0')