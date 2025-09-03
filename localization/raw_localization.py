import cv2
from settings_loader import settings
import numpy as np
from scipy.spatial.transform import Rotation as R

def raw_localization(frames):    
    results = {}
    ret = {}
    for camera_name in settings.cameras:
        camera_index = settings.cameras.index(camera_name)
        ret[camera_name] = False
        
        mtx = np.load(f'{settings.callibration_path}/mtx_{camera_name}.npy')
        dist = np.load(f'{settings.callibration_path}/dist_{camera_name}.npy')
        extrinsic = np.load(f'{settings.callibration_path}/extrinsic_{camera_name}.npy')
            
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        parameters = cv2.aruco.DetectorParameters()
        
        corners, ids, rejected = cv2.aruco.detectMarkers(frames[camera_index], aruco_dict, parameters=parameters)
        
        frames[camera_index] = cv2.aruco.drawDetectedMarkers(frames[camera_index], corners, ids)
        
        if ids is not None:
            ret[camera_name] = True
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, settings.square_size, mtx, dist)

            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R.from_rotvec(rvecs[0]).as_matrix()
            transformation_matrix[:3, 3] = tvecs[0]
            results[camera_name] = np.linalg.inv(extrinsic) @ transformation_matrix
            
    return results, ret