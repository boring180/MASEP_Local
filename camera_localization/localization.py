import cv2
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
sys.path.append(os.path.dirname(os.path.abspath('.')))

from utils.frame_slicing import slicing_frame3_1, slicing_frame3_2
from utils.frame_concatent import concatent_frame3_1, concatent_frame3_2

video_path = '../video/test1.mp4'
callibration_path = '../Callibration/results/'
cameras = ['cam2', 'cam3', 'wide', 'cam0', 'cam1']
SQUARE_SIZE = 0.023 # in meters

def raw_localization(frame):
    frames = slicing_frame3_1(frame)
    frames[0] = cv2.rotate(frames[0], cv2.ROTATE_90_COUNTERCLOCKWISE)
    frames[1] = cv2.rotate(frames[1], cv2.ROTATE_90_CLOCKWISE)
    frames[3] = cv2.rotate(frames[3], cv2.ROTATE_180)
    
    results = {}
    ret = {}
    for camera_name in cameras:
        ret[camera_name] = False
        
        with open(f'{callibration_path}/intrinsic_{camera_name}.json', 'r') as f:
            data = json.load(f)
            mtx = np.array(data['mtx'])
            dist = np.array(data['dist'])

        with open(f'{callibration_path}/extrinsic_{camera_name}.json', 'r') as f:
            data = json.load(f)
            extrinsic = np.array(data)
            
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        parameters = cv2.aruco.DetectorParameters()
        
        corners, ids, rejected = cv2.aruco.detectMarkers(frames[cameras.index(camera_name)], aruco_dict, parameters=parameters)
        
        frames[cameras.index(camera_name)] = cv2.aruco.drawDetectedMarkers(frames[cameras.index(camera_name)], corners, ids)
        
        if ids is not None:
            ret[camera_name] = True
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, SQUARE_SIZE, mtx, dist)

            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R.from_rotvec(rvecs[0]).as_matrix()
            transformation_matrix[:3, 3] = tvecs[0]
            results[camera_name] = np.linalg.inv(extrinsic) @ transformation_matrix
            
    frame = concatent_frame3_1(frames)
            
    return results, frame, ret

def draw_detection_graph(ax, results, ret):
    
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    
    for camera_name in cameras:
        if ret[camera_name]:
            ax.scatter(results[camera_name][0, 0], results[camera_name][1, 0], results[camera_name][2, 0], color=colors[cameras.index(camera_name)])

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FPS, 100)

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1, projection='3d')

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results, frame, ret = raw_localization(frame)
    draw_detection_graph(ax, results, ret)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    count += 1

cap.release()
fig.savefig('detection_graph.png')
plt.show()