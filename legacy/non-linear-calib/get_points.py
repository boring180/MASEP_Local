"""
Extract Calibration Points from Images
Usage: python get_points.py
"""

import cv2
import numpy as np
import json
from pathlib import Path
import tqdm


def main():
    # Load config
    with open("config.json", 'r') as f:
        config = json.load(f)
    
    images_folder = config['images_folder']
    output_file = config['output_file']
    pattern_width = config['pattern_width']
    pattern_height = config['pattern_height']
    square_size = config['square_size']
    
    pattern_size = (pattern_width - 1, pattern_height - 1)
    
    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Storage
    obj_points = []
    img_points = []
    
    # Find images
    img_folder = Path(images_folder)
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        img_files.extend(img_folder.glob(ext))
        img_files.extend(img_folder.glob(ext.upper()))
    img_files = sorted(img_files)
    
    img_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    for img_file in tqdm.tqdm(img_files):
        img = cv2.imread(str(img_file))
        
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            if len(corners) < 6:
                continue
            obj_points.append(objp)
            img_points.append(corners.reshape(-1, 2))
    
    print(f"Detected: {len(obj_points)}/{len(img_files)} images")
    
    np.savez(output_file,
             obj_points=np.array(obj_points),
             img_points=np.array(img_points),
             img_size=np.array(img_size),
             pattern_size=np.array(pattern_size),
             square_size=square_size)
    
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
