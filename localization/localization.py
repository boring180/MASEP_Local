import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import io
import pickle
from settings_loader import settings
from raw_localization import raw_localization
from visualize import visualize
sys.path.append(os.path.dirname(os.path.abspath('.')))

from utils.frame_slicing import slicing_frame3_1, slicing_frame3_2

def main():
    cap = cv2.VideoCapture(settings.video_path)

    # Initialize historical points dictionary
    points = {camera_name: [] for camera_name in settings.cameras}

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames = slicing_frame3_1(frame)
        frames[2] = cv2.rotate(frames[2], cv2.ROTATE_180)
        results, ret_vals = raw_localization(frames)
        
        # Store current points in historical data
        for camera_name in settings.cameras:
            if ret_vals[camera_name]:
                point = [results[camera_name][0, 0], 
                        results[camera_name][1, 0], 
                        results[camera_name][2, 0]]
                points[camera_name].append(point)
        
    if not os.path.exists('output'):
        os.makedirs('output')
    pickle.dump(points, open('output/points.pkl', 'wb'))

    cap.release()

    visualize(points)

if __name__ == '__main__':
    main()