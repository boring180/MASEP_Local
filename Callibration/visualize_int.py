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
from utils.frame_concatent import resize_with_padding

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
    ret, corners, objp = get_points_single_frame(gray, settings, 'intrinsic')
    img = img.copy()
    h,  w = img.shape[:2]
    h_original, w_original = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    dst = resize_with_padding(dst, w_original, h_original)
    
    if ret == True:
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        dst = draw(dst, corners, imgpts)
        return dst
    else:
        return None

def arrow_projection():
    mtxs = {}
    dists = {}
    for camera_name in cameras:
        mtx = np.load(f'results/mtx_{camera_name}.npy')
        dist = np.load(f'results/dist_{camera_name}.npy')
        mtxs[camera_name] = mtx
        dists[camera_name] = dist
            
    fig = plt.figure(figsize=(30, 45))
    
    before_calibration = []
    after_calibration = []

    images = os.listdir(image_path)
    random_index = random.randint(0, len(images) - 1)
    
    while True:
        file_name = image_path + '/' + images[random_index]
        image_before = cv2.imread(file_name)
        if settings.internal_callibration_type == 'single':
            camera_name = file_name.split('/')[-1].split('.')[0].split('_')[0]
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
            
            
    before_calibration = concatent_frame(before_calibration)
    after_calibration = concatent_frame(after_calibration)
    difference = cv2.absdiff(before_calibration, after_calibration)
        
    ax = fig.add_subplot(3, 1, 1)
    ax.imshow(before_calibration)
    ax.set_title(f'Before calibration')
    ax.axis('off')
    
    ax = fig.add_subplot(3, 1, 2)
    ax.imshow(after_calibration)
    ax.set_title(f'After calibration')
    ax.axis('off')
    
    ax = fig.add_subplot(3, 1, 3)
    ax.imshow(difference)
    ax.set_title(f'Difference')
    ax.axis('off')
            
    plt.savefig('results/arrow_projection.png')
    
def main():
    arrow_projection()
    
if __name__ == '__main__':
    main()