import cv2
import numpy as np
import os
import sys

from get_points import get_points

from settings_loader import settings

sys.path.append(os.path.dirname(os.path.abspath('.')))

def camera_intrinsic_calibration(settings):
    cameras = settings.cameras
    
    if os.path.exists('chessboard_points/intrinsic_rets.npy') and os.path.exists('chessboard_points/intrinsic_object_points.npy') and os.path.exists('chessboard_points/intrinsic_image_points.npy') and os.path.exists('chessboard_points/intrinsic_shape.npy'):
        print('Image points already exist')
    else:
        print('Image points not exist, start to get image points')
        get_points(settings, 'intrinsic')
    
    rets = np.load('chessboard_points/intrinsic_rets.npy')
    objpoints = np.load('chessboard_points/intrinsic_object_points.npy')
    imgpoints = np.load('chessboard_points/intrinsic_image_points.npy')
    shape = np.load('chessboard_points/intrinsic_shape.npy')
    
    for camera_name in cameras:
        imgpoints_camera = imgpoints[:, cameras.index(camera_name), :, :]
        objpoints_camera = objpoints[:, cameras.index(camera_name), :, :]
        rets_camera = rets[:, cameras.index(camera_name)]
        imgpoints_camera = imgpoints_camera[rets_camera, :, :]
        objpoints_camera = objpoints_camera[rets_camera, :, :]

        number_of_success_images = np.sum(rets_camera.astype(int))
        
        obj_pts = []
        img_pts = []
        for i in range(number_of_success_images):
            obj_pts.append(objpoints_camera[i, :, :].astype(np.float32))
            img_pts.append(imgpoints_camera[i, :, :].astype(np.float32))
            
        obj_pts = np.array(obj_pts)
        img_pts = np.array(img_pts)        
        
        # flags = cv2.CALIB_RATIONAL_MODEL
        flags = None
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, shape[::-1], None, None, flags=flags)
        
        np.save(f'results/mtx_{camera_name}.npy', mtx)
        np.save(f'results/dist_{camera_name}.npy', dist)
        
        print(f'{camera_name} has {number_of_success_images} successful images')
        
        mean_error = 0
        for i in range(number_of_success_images):
            imgpoints_reprojected, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
            imgpoints_reprojected = imgpoints_reprojected.reshape(-1, 2)
            error = cv2.norm(img_pts[i], imgpoints_reprojected, cv2.NORM_L2) / len(imgpoints_reprojected)
            mean_error += error
        mean_error = mean_error / number_of_success_images
        print(f'{camera_name} has reprojection error: {mean_error}')
            
    return

    
def main():
    os.makedirs('results', exist_ok=True)
    camera_intrinsic_calibration(settings)
    
if __name__ == '__main__':
    main()