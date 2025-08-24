import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import io
from settings_loader import settings
from raw_localization import raw_localization
sys.path.append(os.path.dirname(os.path.abspath('.')))

from utils.frame_slicing import slicing_frame3_1, slicing_frame3_2
from utils.frame_concatent import concatent_frame3_1, concatent_frame3_2

def draw_detection_graph(ax, points):
    
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    
    for camera_name in settings.cameras:
        camera_index = settings.cameras.index(camera_name)
        points_camera = points[camera_name]
        if len(points_camera) == 0:
            continue
        points_camera = np.array(points_camera)
        ax.scatter(points_camera[:, 0], points_camera[:, 1], points_camera[:, 2], 
                    color=colors[camera_index],
                    alpha=0.3, marker='.')

def main():
    cap = cv2.VideoCapture(settings.video_path)
    if not os.path.exists('output'):
        os.makedirs('output')
    filename = f'output/localization.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 24, (1920, 1080))

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
        
        frame_concat = concatent_frame3_1(frames)
        out.write(frame_concat)

    cap.release()
    out.release()
    
    dpi = 100
    fig = plt.figure(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title('3D Point Cloud')
    
    draw_detection_graph(ax, points)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
    plt.close(fig)

if __name__ == '__main__':
    main()