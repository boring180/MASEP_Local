import cv2
import numpy as np
import os

from get_points import get_points
from settings_loader import settings



def extrinsic_calibration(settings):
    if settings.same_pattern:
        if os.path.exists('chessboard_points/intrinsic_rets.npy') and os.path.exists('chessboard_points/intrinsic_object_points.npy') and os.path.exists('chessboard_points/intrinsic_image_points.npy') and os.path.exists('chessboard_points/intrinsic_shape.npy'):
            rets = np.load('chessboard_points/intrinsic_rets.npy')
            objpoints = np.load('chessboard_points/intrinsic_object_points.npy')
            imgpoints = np.load('chessboard_points/intrinsic_image_points.npy')
            shape = np.load('chessboard_points/intrinsic_shape.npy')
    else:
        if os.path.exists('chessboard_points/extrinsic_rets.npy') and os.path.exists('chessboard_points/extrinsic_object_points.npy') and os.path.exists('chessboard_points/extrinsic_image_points.npy') and os.path.exists('chessboard_points/extrinsic_shape.npy'):
            pass
        else:
            print('Image points not exist, start to get image points')
            get_points(settings, calibration_type='extrinsic')
            
        rets = np.load('chessboard_points/extrinsic_rets.npy')
        objpoints = np.load('chessboard_points/extrinsic_object_points.npy')
        imgpoints = np.load('chessboard_points/extrinsic_image_points.npy')
        shape = np.load('chessboard_points/extrinsic_shape.npy')
    
    cameras = settings.cameras
    center_camera = settings.center_camera
    
    mtxs = {}
    dists = {}
    for camera_name in settings.cameras:
        mtxs[camera_name] = np.load(f'results/mtx_{camera_name}.npy')
        dists[camera_name] = np.load(f'results/dist_{camera_name}.npy')
    
    for camera_name in cameras:
        transformation_matrix = np.eye(4)
        if camera_name == center_camera:
            pass
        else:
            stereo_objpoints = []
            stereo_imgpoints1 = []
            stereo_imgpoints2 = []
            for i in range(len(objpoints)):
                if rets[i, cameras.index(camera_name)] and rets[i, cameras.index(center_camera)]:
                    stereo_objpoints.append(objpoints[i, cameras.index(camera_name), :, :].astype(np.float32))
                    stereo_imgpoints1.append(imgpoints[i, cameras.index(camera_name), :, :].astype(np.float32))
                    stereo_imgpoints2.append(imgpoints[i, cameras.index(center_camera), :, :].astype(np.float32))
            stereo_objpoints = np.array(stereo_objpoints)
            stereo_imgpoints1 = np.array(stereo_imgpoints1)
            stereo_imgpoints2 = np.array(stereo_imgpoints2)
            
            flags = (cv2.CALIB_FIX_INTRINSIC)
            results = cv2.stereoCalibrate(objectPoints=stereo_objpoints, imagePoints1=stereo_imgpoints1, imagePoints2=stereo_imgpoints2,
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
            
            
        print(f'{camera_name} transformation matrix: {transformation_matrix}')
        with open(f'results/extrinsic_calibration.log', 'a') as f:
            f.write(f'{camera_name} transformation matrix: {transformation_matrix}\n')

        np.save(f'results/extrinsic_{camera_name}.npy', transformation_matrix)
    
    
def main():   
    extrinsic_calibration(settings)
    
if __name__ == '__main__':
    main()