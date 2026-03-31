"""
Extract Calibration Points from ChArUco Board Images
Usage: python get_points_charuco.py
"""

import cv2
import numpy as np
import json
from pathlib import Path
import tqdm


def main():
    # Load config
    with open("config_charuco.json", 'r') as f:
        config = json.load(f)
    
    images_folder = config['images_folder']
    output_file = config['output_file']
    aruco_dict_name = config['aruco_dict']
    charuco_squares_x = config['charuco_squares_x']
    charuco_squares_y = config['charuco_squares_y']
    square_size = config['square_size']
    marker_size = config['marker_size']
    
    # Create ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    board = cv2.aruco.CharucoBoard(
        (charuco_squares_x, charuco_squares_y),
        square_size,
        marker_size,
        aruco_dict
    )
    
    # Create detector
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Storage
    all_charuco_corners = []
    all_charuco_ids = []
    
    # Find images
    img_folder = Path(images_folder)
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        img_files.extend(img_folder.glob(ext))
        img_files.extend(img_folder.glob(ext.upper()))
    img_files = sorted(img_files)
    
    img_size = None
    
    print(f"Processing {len(img_files)} images...")
    
    obj_points_list = []
    img_points_list = []
    
    for img_file in tqdm.tqdm(img_files):
        img = cv2.imread(str(img_file))
        
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        marker_corners, marker_ids, rejected = detector.detectMarkers(gray)
        
        # If at least one marker is detected
        if marker_ids is not None and len(marker_ids) > 0:
            # Interpolate ChArUco corners
            num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )
            
            # If enough corners are found, add to the list
            if num_corners > 15:  # Need at least 6 corners for calibration
                obj_points = board.getChessboardCorners()[charuco_ids.flatten()]
                img_points = charuco_corners.reshape(-1, 2)
                
                obj_points_list.append(obj_points)
                img_points_list.append(img_points)
    
    print(f"Detected objp in: {len(obj_points_list)}/{len(img_files)} images")
    
    if len(obj_points_list) == 0:
        print("ERROR: No ChArUco boards detected!")
        return
    
    # Save data
    np.savez(output_file,
             obj_points=np.array(obj_points_list, dtype=object),
             img_points=np.array(img_points_list, dtype=object),
             img_size=np.array(img_size),
             charuco_ids=np.array(all_charuco_ids, dtype=object),
             board_size=(charuco_squares_x, charuco_squares_y),
             square_size=square_size,
             marker_size=marker_size)
    
    print(f"Saved: {output_file}")
    
    # Print statistics on FILTERED data
    corners_per_image = [len(pts) for pts in img_points_list]
    print(f"\nStatistics (saved data):")
    print(f"  Images: {len(obj_points_list)}")
    print(f"  Min corners per image: {min(corners_per_image)}")
    print(f"  Max corners per image: {max(corners_per_image)}")
    print(f"  Avg corners per image: {np.mean(corners_per_image):.1f}")


if __name__ == "__main__":
    main()
