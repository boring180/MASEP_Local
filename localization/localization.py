import cv2
import numpy as np
import os
import sys
import tqdm

from settings_loader import settings
from raw_localization import raw_localization
from visualize import visualize
sys.path.append(os.path.dirname(os.path.abspath('.')))

from utils.frame_slicing import slicing_frame3_1, slicing_frame3_2

def main():
    cap = cv2.VideoCapture(settings.video_path)

    # Initialize historical points dictionary
    points = []
    rets = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for _ in tqdm.tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            continue
        
        frames = slicing_frame3_1(frame)
        frames[2] = cv2.rotate(frames[2], cv2.ROTATE_180)
        results, ret_vals = raw_localization(frames)
        
        frame_points = [[] for i in range(len(settings.cameras))]
        frame_rets = [[] for i in range(len(settings.cameras))]
        # Store current points in historical data
        for camera_name in settings.cameras:
            if ret_vals[camera_name]:
                frame_rets[settings.cameras.index(camera_name)] = True
                # frame_points[settings.cameras.index(camera_name)] = results[camera_name][:3, 3]
                frame_points[settings.cameras.index(camera_name)] = results[camera_name]
            else:
                frame_rets[settings.cameras.index(camera_name)] = False
                frame_points[settings.cameras.index(camera_name)] = np.zeros((3,)).astype(float)
                
        points.append(frame_points)
        rets.append(frame_rets)
        
    if not os.path.exists('output'):
        os.makedirs('output')
        
    points = np.array(points)
    rets = np.array(rets)
    np.save('output/points.npy', points)
    np.save('output/rets.npy', rets)

    cap.release()

    visualize()

if __name__ == '__main__':
    main()