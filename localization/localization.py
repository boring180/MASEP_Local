import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from settings_loader import settings
from raw_localization import raw_localization
sys.path.append(os.path.dirname(os.path.abspath('.')))

from utils.frame_slicing import slicing_frame3_1, slicing_frame3_2
from utils.frame_concatent import concatent_frame3_1, concatent_frame3_2

def draw_detection_graph(ax, results, ret):
    
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    
    for camera_name in settings.cameras:
        if ret[camera_name]:
            ax.scatter(results[camera_name][0, 0], results[camera_name][1, 0], results[camera_name][2, 0], color=colors[settings.cameras.index(camera_name)])

def main():
    cap = cv2.VideoCapture(settings.video_path)
    if not os.path.exists('output'):
        os.makedirs('output')
    filename = f'output/localization.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 24, (1920, 1080))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fig = plt.figure(figsize=(6, 4))
        
        ax = fig.add_subplot(1, 2, 1)
        results, ret = raw_localization(frame)
        ax.imshow(frame)
        
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        draw_detection_graph(ax, results, ret, frame)
        
        out.write(frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    main()