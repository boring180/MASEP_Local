import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import sys
import time
import random
random.seed(time.time())

sys.path.append(os.path.dirname(os.path.abspath('.')))
import utils.frame_slicing as frame_slicing
import utils.frame_concatent as frame_concatent

from get_points import get_points_single_frame

cameras = ['cam2', 'cam3', 'wide', 'cam0', 'cam1']

image_path = '../photos/single_camera'

number_of_squares_x = 36
number_of_squares_y = 14
number_of_internal_corners_x = number_of_squares_x - 1
number_of_internal_corners_y = number_of_squares_y - 1
square_size = 5.4/6.0  # in meters

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axis = axis * square_size

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def projection(img, mtx, dist):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objp = np.zeros((number_of_internal_corners_x * number_of_internal_corners_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:number_of_internal_corners_x,0:number_of_internal_corners_y].T.reshape(-1,2)
    objp = objp * square_size
    ret, corners, objp = get_points_single_frame(gray, number_of_internal_corners_x, number_of_internal_corners_y, objp)
    if ret == True:
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img, corners, imgpts)
        
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

    for i in range(len(cameras)):
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
            
    plt.savefig('results/arrow_projection.png')
    
def main():
    arrow_projection()
    
if __name__ == '__main__':
    main()