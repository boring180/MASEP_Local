import cv2
import numpy as np
import glob
import json
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import sys
import time
import random
random.seed(time.time())

sys.path.append(os.path.dirname(os.path.abspath('.')))
import utils.frame_slicing as frame_slicing
import utils.frame_concatent as frame_concatent

number_of_squares_x = 11
number_of_internal_corners_x = number_of_squares_x - 1
number_of_squares_y = 8
number_of_internal_corners_y = number_of_squares_y - 1
SQUARE_SIZE = 0.023 # in meters
cameras = ['cam2', 'cam3', 'wide', 'cam0', 'cam1']

image_path = '../photos/multi_camera'

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(10,9,0)
objp = np.zeros((number_of_internal_corners_x * number_of_internal_corners_y,3), np.float32)
objp[:,:2] = np.mgrid[0:number_of_internal_corners_x,0:number_of_internal_corners_y].T.reshape(-1,2)
objp = objp * SQUARE_SIZE

def get_image_points():
    imgpoints = [] # 2D points in image plane
    objpoints = [] # 3D points in world coordinate system
    rets = [] # boolean values
    
    images = glob.glob(f'{image_path}/*.jpg')
    
    os.makedirs('chessboard_points', exist_ok=True)
    
    shape = (0,0)
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        frames = frame_slicing.slicing_frame(gray)[:5]
        
        frame_imgpoints = np.zeros((len(frames), number_of_internal_corners_x * number_of_internal_corners_y, 2))
        frame_objpoints = np.zeros((len(frames), number_of_internal_corners_x * number_of_internal_corners_y, 3))
        frame_rets = np.zeros(len(frames), dtype=bool) + True
        
        for i in range(len(frames)):
            frame = frames[i]
            shape = frame.shape
            
            ret, corners = cv2.findChessboardCorners(frame, (number_of_internal_corners_x,number_of_internal_corners_y), None)
            
            if ret == True:
                corners2 = cv2.cornerSubPix(frame, corners, (11,11), (-1,-1), criteria=criteria)
                frame_objpoints[i, :, :] = objp
                frame_imgpoints[i, :, :] = corners2[:, 0, :]
                
                cv2.drawChessboardCorners(frame, (number_of_internal_corners_x,number_of_internal_corners_y), corners2, ret)
                
            else:
                frame_rets[i] = False
                
        imgpoints.append(frame_imgpoints)
        objpoints.append(frame_objpoints)
        rets.append(frame_rets)
        
        frame_with_corners = frame_concatent.concatent_frame(frames)
        cv2.imwrite(f'chessboard_points/{fname.split("/")[-1]}', frame_with_corners)
        
    imgpoints = np.array(imgpoints)
    objpoints = np.array(objpoints)
    rets = np.array(rets)
    
    os.makedirs('results', exist_ok=True)
        
    if rets.sum() == 0:
        print('No chessboard found in any image')
        return
    
    with open('results/rets.json', 'w') as f:
        json.dump(rets.tolist(), f)
    with open('results/object_points.json', 'w') as f:
        json.dump(objpoints.tolist(), f)
    with open('results/image_points.json', 'w') as f:
        json.dump(imgpoints.tolist(), f)
    with open('results/shape.json', 'w') as f:
        json.dump(shape, f)
        
def intrinsic_calibration(cameras):
    if not os.path.exists('results/rets.json') or not os.path.exists('results/object_points.json') or not os.path.exists('results/image_points.json') or not os.path.exists('results/shape.json'):
        get_image_points()
    
    with open('results/rets.json', 'r') as f:
        rets = np.array(json.load(f))
    with open('results/object_points.json', 'r') as f:
        objpoints = np.array(json.load(f))
    with open('results/image_points.json', 'r') as f:
        imgpoints = np.array(json.load(f))
    with open('results/shape.json', 'r') as f:
        shape = tuple(json.load(f))
    
    for camera_name in cameras:
        imgpoints_camera = imgpoints[:, cameras.index(camera_name), :, :]
        objpoints_camera = objpoints[:, cameras.index(camera_name), :, :]
        rets_camera = rets[:, cameras.index(camera_name)]
        imgpoints_camera = imgpoints_camera[rets_camera, :, :]
        objpoints_camera = objpoints_camera[rets_camera, :, :]
        
        number_of_images = len(rets_camera)
        number_of_success_images = np.sum(rets_camera.astype(int))
        
        obj_pts = []
        img_pts = []
        for i in range(number_of_success_images):
            obj_pts.append(objpoints_camera[i, :, :].astype(np.float32))
            img_pts.append(imgpoints_camera[i, :, :].astype(np.float32))
            
        obj_pts = np.array(obj_pts)
        img_pts = np.array(img_pts)        
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, shape[::-1], None, None)
        
        with open(f'results/intrinsic_{camera_name}.json', 'w') as f:
            json.dump({'mtx': mtx.tolist(), 'dist': dist.tolist()}, f)
        
        print(f'{camera_name} has {number_of_success_images}/{number_of_images} successful images')
        
        mean_error = 0
        for i in range(number_of_success_images):
            imgpoints2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
            # Reshape imgpoints2 to match img_pts dimensions
            imgpoints2 = imgpoints2.reshape(-1, 2)
            error = cv2.norm(img_pts[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        mean_error = mean_error / number_of_success_images
        print(f'{camera_name} has mean error: {mean_error}')
            
    return

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axis = axis * SQUARE_SIZE

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def projection(img, mtx, dist):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (number_of_internal_corners_x, number_of_internal_corners_y), None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria=criteria)
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img, corners2, imgpts)
        
    return img

def arrow_projection():
    mtxs = {}
    dists = {}
    for camera_name in cameras:
        with open(f'results/intrinsic_{camera_name}.json', 'r') as f:
            data = json.load(f)
            mtx = np.array(data['mtx'])
            dist = np.array(data['dist'])
            mtxs[camera_name] = mtx
            dists[camera_name] = dist
            
    fig = plt.figure(figsize=(20, 25))

    for i in range(5):
        images = os.listdir(image_path)
        random_index = random.randint(0, len(images) - 1)
        image = cv2.imread(f'{image_path}/{images[random_index]}')
        ax = fig.add_subplot(5, 2, i * 2 + 1)
        ax.imshow(image)
        ax.set_title(f'Before calibration')
        ax.axis('off')
        
        frames = frame_slicing.slicing_frame(image)[:5]
        
        for camera_name in cameras:
            frame_index = cameras.index(camera_name)
            frame = frames[frame_index] 
            dst = cv2.undistort(frame, mtxs[camera_name], dists[camera_name])
            frames[cameras.index(camera_name)] = projection(dst, mtxs[camera_name], dists[camera_name])
        
        images_after_calibration = frame_concatent.concatent_frame(frames)
        ax = fig.add_subplot(5, 2, i * 2 + 2)
        ax.imshow(images_after_calibration)
        ax.set_title(f'After calibration')
        ax.axis('off')
            
    plt.savefig('arrow_projection.png')
    
def main():
    if os.path.exists('results/rets.json') and os.path.exists('results/object_points.json') and os.path.exists('results/image_points.json') and os.path.exists('results/shape.json'):
        pass
    else:
        get_image_points()
            
    results = os.listdir('results')
    if 'intrinsic_cam0.json' in results and 'intrinsic_cam1.json' in results and 'intrinsic_cam2.json' in results and 'intrinsic_cam3.json' in results and 'intrinsic_wide.json' in results:
        pass
    else:
        intrinsic_calibration(cameras)
        
    arrow_projection()
    
if __name__ == '__main__':
    main()