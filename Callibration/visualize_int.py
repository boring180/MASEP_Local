import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import sys
import time
import random
random.seed(time.time())

from settings_loader import settings

sys.path.append(os.path.dirname(os.path.abspath('.')))
from utils.frame_slicing import slicing_frame
from utils.frame_concatent import concatent_frame

from get_points import get_points_single_frame

# Use settings from configuration file
cameras = settings.cameras
# Automatically determine image path based on calibration type
if settings.internal_callibration_type == 'single':
    image_path = '../photos/single_camera'
else:
    image_path = '../photos/multi_camera'
number_of_internal_corners_x = settings.pattern_size_internal[0]
number_of_internal_corners_y = settings.pattern_size_internal[1]
square_size = settings.pattern_square_size_internal

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
    ret, corners, objp = get_points_single_frame(gray, settings)
    if ret == True:
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img, corners, imgpts)
        return img
    else:
        return None

def arrow_projection():
    mtxs = {}
    dists = {}
    for camera_name in cameras:
        data = np.load(f'results/intrinsic_{camera_name}.npy')
        mtx = data[0]
        dist = data[1]
        mtxs[camera_name] = mtx
        dists[camera_name] = dist
            
    fig = plt.figure(figsize=(20, 15))
    
    before_calibration = []
    after_calibration = []

    images = os.listdir(image_path)
    random_index = random.randint(0, len(images) - 1)
    
    while True:
        image_before = cv2.imread(f'{image_path}/{images[random_index]}')
        if settings.internal_callibration_type == 'single':
            camera_name = image_before.split('/')[-1].split('.')[0].split('_')[0]
            image_after = projection(image_before, mtxs[camera_name], dists[camera_name])
            if image_after is not None:
                before_calibration.append(image_before)
                after_calibration.append(image_after)
        else:
            frames = slicing_frame(image_before)[:5]
            for camera_name in cameras:
                frame_index = cameras.index(camera_name)
                frame_before = frames[frame_index]
                frame_after = projection(frame_before, mtxs[camera_name], dists[camera_name])
                if frame_after is not None:
                    before_calibration.append(frame_before)
                    after_calibration.append(frame_after)
        
        if len(before_calibration) == len(settings.cameras):
            break
        else:
            random_index = random.randint(0, len(images) - 1)
        
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(concatent_frame(before_calibration))
    ax.set_title(f'Before calibration')
    ax.axis('off')
    
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(concatent_frame(after_calibration))
    ax.set_title(f'After calibration')
    ax.axis('off')
            
    plt.savefig('results/arrow_projection.png')
    
def main():
    arrow_projection()
    
if __name__ == '__main__':
    main()