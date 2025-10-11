import cv2
import numpy as np
import os
import sys
import pickle

from get_points import get_points

from settings_loader import settings

sys.path.append(os.path.dirname(os.path.abspath('.')))

def camera_intrinsic_calibration(settings):
    cameras = settings.cameras
    
    if os.path.exists('chessboard_points/intrinsic_rets.pkl') and os.path.exists('chessboard_points/intrinsic_object_points.pkl') and os.path.exists('chessboard_points/intrinsic_image_points.pkl') and os.path.exists('chessboard_points/intrinsic_shape.pkl'):
        print('Image points already exist')
    else:
        print('Image points not exist, start to get image points')
        get_points(settings, 'intrinsic')
    
    rets = pickle.load(open('chessboard_points/intrinsic_rets.pkl', 'rb'))
    objpoints = pickle.load(open('chessboard_points/intrinsic_object_points.pkl', 'rb'))
    imgpoints = pickle.load(open('chessboard_points/intrinsic_image_points.pkl', 'rb'))
    shape = pickle.load(open('chessboard_points/intrinsic_shape.pkl', 'rb'))
    number_of_images = len(rets)
    
    for camera_name in cameras:
        camera_obj_pts = []
        camera_img_pts = []
        number_of_success_images = 0
        for i in range(number_of_images):
            if rets[i][camera_name] == True:
                number_of_success_images += 1
                camera_obj_pts.append(objpoints[i][camera_name].astype(np.float32))
                camera_img_pts.append(imgpoints[i][camera_name].astype(np.float32))
                
        print(f'{camera_name} has {number_of_success_images} successful images')
        with open(f'results/intrinsic_calibration.log', 'a') as f:
            f.write(f'{camera_name} has {number_of_success_images} successful images\n')
            
        camera_obj_pts = np.array(camera_obj_pts)
        camera_img_pts = np.array(camera_img_pts)        
        
        # flags = cv2.CALIB_RATIONAL_MODEL
        # flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL
        termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        flags = None
        ret, mtx, dist, rvecs, tvecs, _, _, error = cv2.calibrateCameraExtended(camera_obj_pts, camera_img_pts, shape[::-1], None, None, flags=flags, criteria=termination_criteria)
        error = np.mean(error)
        pickle.dump(mtx, open(f'results/mtx_{camera_name}.pkl', 'wb'))
        pickle.dump(dist, open(f'results/dist_{camera_name}.pkl', 'wb'))
        
        print(f'{camera_name} has reprojection error: {error}')
        with open(f'results/intrinsic_calibration.log', 'a') as f:
            f.write(f'{camera_name} has reprojection error: {error}\n')
    return
    
def main():
    os.makedirs('results', exist_ok=True)
    
    
if __name__ == '__main__':
    main() 