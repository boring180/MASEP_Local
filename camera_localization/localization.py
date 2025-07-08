import cv2
import numpy as np
import os
import sys
import json
from scipy.spatial.transform import Rotation as R
sys.path.append(os.path.dirname(os.path.abspath('.')))

import utils.frame_slicing as frame_slicing
import utils.frame_concatent as frame_concatent

video_path = '../video/test1.mp4'
callibration_path = '../Callibration/results/'
cameras = ['cam2', 'cam3', 'wide', 'cam0', 'cam1']
SQUARE_SIZE = 0.023 # in meters

def raw_localization(frame):
    frames = frame_slicing.slicing_frame(frame)
    mtxs = {}
    dists = {}
    results = {}
    for camera_name in cameras:
        with open(f'{callibration_path}/intrinsic_{camera_name}.json', 'r') as f:
            data = json.load(f)
            mtx = np.array(data['mtx'])
            dist = np.array(data['dist'])
            mtxs[cameras[cameras.index(camera_name)]] = mtx
            dists[cameras[cameras.index(camera_name)]] = dist
            
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        parameters = cv2.aruco.DetectorParameters()
        
        corners, ids, rejected = cv2.aruco.detectMarkers(frames[cameras.index(camera_name)], aruco_dict, parameters=parameters)
        
        frames[cameras.index(camera_name)] = cv2.aruco.drawDetectedMarkers(frames[cameras.index(camera_name)], corners, ids)
        
        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, SQUARE_SIZE, mtx, dist)

            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R.from_rotvec(rvecs[0]).as_matrix()
            transformation_matrix[:3, 3] = tvecs[0]
            results[cameras[cameras.index(camera_name)]] = transformation_matrix
            
    frame = frame_concatent.concatent_frame(frames)
            
    return results, frame

time_stamp = 0
cap = cv2.VideoCapture(video_path)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results, frame = raw_localization(frame)
    print(results)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    time_stamp += 1
    count += 1

cap.release()