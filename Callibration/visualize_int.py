import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import sys
import time
import random
import pickle
random.seed(time.time())

from settings_loader import settings

sys.path.append(os.path.dirname(os.path.abspath('.')))
from utils.frame_slicing import slicing_frame3_1, slicing_frame3_2
from utils.frame_concatent import concatent_frame3_2, concatent_frame3_1
from utils.frame_concatent import resize_with_padding
from get_points import get_points_single_frame


### ----------------------------- Initialize variables ----------------------------- ###
cameras = settings.cameras
if settings.internal_callibration_type == 'single':
    image_path = '../photos/single_camera'
else:
    image_path = '../photos/multi_camera'

### ----------------------------- Undistort function ----------------------------- ###
def undistort(img, mtx, dist):
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def undistort_comparison():
    mtxs = {}
    dists = {}
    for camera_name in cameras:
        mtx = pickle.load(open(f'results/mtx_{camera_name}.pkl', 'rb'))
        dist = pickle.load(open(f'results/dist_{camera_name}.pkl', 'rb'))
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
        image_before = cv2.cvtColor(image_before, cv2.COLOR_BGR2GRAY)
        if settings.internal_callibration_type == 'single':
            camera_name = file_name.split('/')[-1].split('.')[0].split('_')[0]
            if camera_name not in settings.cameras:
                random_index = random.randint(0, len(images) - 1)
                continue
            image_after = undistort(image_before.copy(), mtxs[camera_name], dists[camera_name])
            image_after = resize_with_padding(image_after, image_before.shape[1], image_before.shape[0])
            before_calibration.append(image_before)
            after_calibration.append(image_after)
            
        else:
            frames = slicing_frame3_1(image_before)
            for camera_name in cameras:
                frame_index = cameras.index(camera_name)
                frame_before = frames[frame_index]
                frame_after = undistort(frame_before.copy(), mtxs[camera_name], dists[camera_name])
                frame_after = resize_with_padding(frame_after, frame_before.shape[1], frame_before.shape[0])
                before_calibration.append(frame_before)
                after_calibration.append(frame_after)
        
        if len(before_calibration) == len(settings.cameras):
            break
        else:
            random_index = random.randint(0, len(images) - 1)
        
    
    before_calibration = concatent_frame3_1(before_calibration)
    after_calibration = concatent_frame3_1(after_calibration)
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
            
    plt.savefig('results/distortion_comparison.png')
    
def main():
    undistort_comparison()
    
if __name__ == '__main__':
    main()