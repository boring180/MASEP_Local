import cv2
import numpy as np
import json
import os

from get_points import get_points

cameras = ['cam2', 'cam3', 'wide', 'cam0', 'cam1']
image_path = '../photos/multi_camera'

def extrinsic_calibration():
    if os.path.exists('chessboard_points/rets.json') and os.path.exists('chessboard_points/object_points.json') and os.path.exists('chessboard_points/image_points.json') and os.path.exists('chessboard_points/shape.json'):
        pass
    else:
        print('Image points not exist, start to get image points')
        get_points(image_path)
    
    with open('chessboard_points/rets.json', 'r') as f:
        rets = np.array(json.load(f))
    with open('chessboard_points/object_points.json', 'r') as f:
        objpoints = np.array(json.load(f))
    with open('chessboard_points/image_points.json', 'r') as f:
        imgpoints = np.array(json.load(f))
    with open('chessboard_points/shape.json', 'r') as f:
        shape = tuple(json.load(f))
    mtxs = {}
    dists = {}
    for camera_name in cameras:
        with open(f'results/intrinsic_{camera_name}.json', 'r') as f:
            data = json.load(f)
            mtx = np.array(data['mtx'])
            dist = np.array(data['dist'])
            mtxs[camera_name] = mtx
            dists[camera_name] = dist
    
    for camera_name in cameras:
        transformation_matrix = np.eye(4)
        if camera_name == 'wide':
            pass
        else:
            stereo_objpoints = []
            stereo_imgpoints1 = []
            stereo_imgpoints2 = []
            for i in range(len(objpoints)):
                if rets[i, cameras.index(camera_name)] and rets[i, cameras.index('wide')]:
                    stereo_objpoints.append(objpoints[i, cameras.index(camera_name), :, :].astype(np.float32))
                    stereo_imgpoints1.append(imgpoints[i, cameras.index(camera_name), :, :].astype(np.float32))
                    stereo_imgpoints2.append(imgpoints[i, cameras.index('wide'), :, :].astype(np.float32))
            stereo_objpoints = np.array(stereo_objpoints)
            stereo_imgpoints1 = np.array(stereo_imgpoints1)
            stereo_imgpoints2 = np.array(stereo_imgpoints2)
            
            flags = (cv2.CALIB_FIX_INTRINSIC)
            results = cv2.stereoCalibrate(objectPoints=stereo_objpoints, imagePoints1=stereo_imgpoints1, imagePoints2=stereo_imgpoints2,
                                        cameraMatrix1=mtxs[camera_name], distCoeffs1=dists[camera_name], cameraMatrix2=mtxs['wide'], 
                                        distCoeffs2=dists['wide'], imageSize=shape, 
                                        flags=flags)
            R = results[5]
            T = results[6]
            transformation_matrix[:3, :3] = R
            transformation_matrix[:3, 3] = T[:3, 0]
        
            print(f'{camera_name} has {len(stereo_objpoints)} successful images')
            
            
        print(f'{camera_name} transformation matrix: {transformation_matrix}')
        
        with open(f'results/extrinsic_{camera_name}.json', 'w') as f:
            json.dump(transformation_matrix.tolist(), f)
    
    
def main():   
    if os.path.exists('results/intrinsic_cam0.json') and os.path.exists('results/intrinsic_cam1.json') and os.path.exists('results/intrinsic_cam2.json') and os.path.exists('results/intrinsic_cam3.json') and os.path.exists('results/intrinsic_wide.json'):
        pass
    else:
        print('Intrinsic calibration not done, please run intrinsic_calibration.sh first')
        return
        
    if os.path.exists('results/extrinsic_cam0.json') and os.path.exists('results/extrinsic_cam1.json') and os.path.exists('results/extrinsic_cam2.json') and os.path.exists('results/extrinsic_cam3.json') and os.path.exists('results/extrinsic_wide.json'):
        pass
    else:
        print('Extrinsic calibration not done, start to do extrinsic calibration')
        extrinsic_calibration()
    
if __name__ == '__main__':
    main()