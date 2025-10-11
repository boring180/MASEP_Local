import cv2
import numpy as np
import os
import pickle

from get_points import get_points
from settings_loader import settings

def extrinsic_calibration(settings):
    if settings.same_pattern:
        if os.path.exists('chessboard_points/intrinsic_rets.pkl') and os.path.exists('chessboard_points/intrinsic_object_points.pkl') and os.path.exists('chessboard_points/intrinsic_image_points.pkl') and os.path.exists('chessboard_points/intrinsic_shape.pkl'):
            rets = pickle.load(open('chessboard_points/intrinsic_rets.pkl', 'rb'))
            objpoints = pickle.load(open('chessboard_points/intrinsic_object_points.pkl', 'rb'))
            imgpoints = pickle.load(open('chessboard_points/intrinsic_image_points.pkl', 'rb'))
            shape = pickle.load(open('chessboard_points/intrinsic_shape.pkl', 'rb'))
    else:
        if os.path.exists('chessboard_points/extrinsic_rets.pkl') and os.path.exists('chessboard_points/extrinsic_object_points.pkl') and os.path.exists('chessboard_points/extrinsic_image_points.pkl') and os.path.exists('chessboard_points/extrinsic_shape.pkl'):
            pass
        else:
            print('Image points not exist, start to get image points')
            get_points(settings, calibration_type='extrinsic')
            
        rets = pickle.load(open('chessboard_points/extrinsic_rets.pkl', 'rb'))
        objpoints = pickle.load(open('chessboard_points/extrinsic_object_points.pkl', 'rb'))
        imgpoints = pickle.load(open('chessboard_points/extrinsic_image_points.pkl', 'rb'))
        shape = pickle.load(open('chessboard_points/extrinsic_shape.pkl', 'rb'))
    
    cameras = settings.cameras
    center_camera = settings.center_camera
    
    mtxs = {}
    dists = {}
    for camera_name in settings.cameras:
        mtxs[camera_name] = pickle.load(open(f'results/mtx_{camera_name}.pkl', 'rb'))
        dists[camera_name] = pickle.load(open(f'results/dist_{camera_name}.pkl', 'rb'))
    
    for camera_name in cameras:
        transformation_matrix = np.eye(4)
        if camera_name == center_camera:
            pass
        else:
            stereo_objpoints_camera = []
            stereo_objpoints_center = []
            stereo_imgpoints_camera = []
            stereo_imgpoints_center = []
            for i in range(len(objpoints)):
                if rets[i][camera_name] and rets[i][center_camera]:
                    stereo_objpoints_camera.append(objpoints[i][camera_name].astype(np.float32))
                    stereo_objpoints_center.append(objpoints[i][center_camera].astype(np.float32))
                    stereo_imgpoints_camera.append(imgpoints[i][camera_name].astype(np.float32))
                    stereo_imgpoints_center.append(imgpoints[i][center_camera].astype(np.float32))
                    
                    
            stereo_objpoints_camera = np.array(stereo_objpoints_camera)
            stereo_objpoints_center = np.array(stereo_objpoints_center)
            stereo_imgpoints_camera = np.array(stereo_imgpoints_camera)
            stereo_imgpoints_center = np.array(stereo_imgpoints_center)
            
            flags = (cv2.CALIB_FIX_INTRINSIC)
            results = cv2.stereoCalibrate(objectPoints=stereo_objpoints, imagePoints1=stereo_imgpoints_camera, imagePoints2=stereo_imgpoints_center,
                                        cameraMatrix1=mtxs[camera_name], distCoeffs1=dists[camera_name], cameraMatrix2=mtxs[center_camera], 
                                        distCoeffs2=dists[center_camera], imageSize=shape, 
                                        flags=flags)
            R = results[5]
            T = results[6]
            transformation_matrix[:3, :3] = R
            transformation_matrix[:3, 3] = T[:3, 0]
        
            print(f'{camera_name} has {len(stereo_objpoints)} successful images')
            with open(f'results/extrinsic_calibration.log', 'a') as f:
                f.write(f'{camera_name} has {len(stereo_objpoints)} successful images\n')
                
            number_of_success_images = stereo_objpoints.shape[0]
            reprojection_errors = []
            for i in range(number_of_success_images):
                ret_camera, tvecs_camera, rvecs_camera = cv2.solvePnP(stereo_objpoints[i, :, :], stereo_imgpoints_camera[i, :, :], mtxs[camera_name], dists[camera_name])
                ret_center, tvecs_center, rvecs_center = cv2.solvePnP(stereo_objpoints[i, :, :], stereo_imgpoints_center[i, :, :], mtxs[center_camera], dists[center_camera])
                if ret_camera == True and ret_center == True:
                    error = tvecs_camera - tvecs_center
                    reprojection_errors.append(error)
            
            mean_reprojection_error = np.mean(reprojection_errors, axis=0)
            std_reprojection_error = np.std(reprojection_errors, axis=0)
            print(f'{camera_name} has reprojection error mean: {mean_reprojection_error}')
            print(f'{camera_name} has reprojection error std: {std_reprojection_error}')
            with open(f'results/extrinsic_calibration.log', 'a') as f:
                f.write(f'{camera_name} has reprojection error mean: {mean_reprojection_error}\n')
                f.write(f'{camera_name} has reprojection error std: {std_reprojection_error}\n')
                    
            
            
    
        print(f'{camera_name} transformation matrix: {transformation_matrix}')
        with open(f'results/extrinsic_calibration.log', 'a') as f:
            f.write(f'{camera_name} transformation matrix: {transformation_matrix}\n')

        pickle.dump(transformation_matrix, open(f'results/extrinsic_{camera_name}.pkl', 'wb'))
    
    
def main():   
    extrinsic_calibration(settings)
    
if __name__ == '__main__':
    main()