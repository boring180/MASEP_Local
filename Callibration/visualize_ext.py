import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import sys
import time
import random
random.seed(time.time())
from get_points import get_points_single_frame
from settings_loader import settings

sys.path.append(os.path.dirname(os.path.abspath('.')))
from utils.frame_slicing import slicing_frame3_1, slicing_frame3_2
image_path = '../photos/multi_camera'

def visualize(settings, fig, elev, azim, roll, i):
    transformation_matrices = {}
    for camera_name in settings.cameras:
        transformation_matrix = np.load(f'results/extrinsic_{camera_name}.npy')
        transformation_matrices[camera_name] = transformation_matrix
            
    ax = fig.add_subplot(1, 3, i, projection='3d', elev=elev, azim=azim, roll=roll)
    length = 0.5
    width = 0.5
    height = 0.5
    
    x = [-length/2, length/2, length/2, -length/2]
    y = [-width/2, -width/2, width/2, width/2]
    for i in range(4):
        ax.plot([x[i], x[i]], [y[i], y[i]], [0, height], 'b-')
        
    x = [-length/2, length/2, length/2, -length/2]
    z = [0, 0, height, height]
    for i in range(4):
        ax.plot([x[i], x[i]], [-width/2, width/2], [z[i], z[i]], 'b-')
        
    y = [-length/2, length/2, length/2, -length/2]
    z = [0, 0, height, height]
    for i in range(4):
        ax.plot([-length/2, length/2], [y[i], y[i]], [z[i], z[i]], 'b-')
    
    for camera_name in settings.cameras:
        colors = ['red', 'green', 'blue']
        axis = np.eye(3) * settings.pattern_square_size_external
        for i in range(3):
            orientation = transformation_matrices[camera_name][:3, :3] @ axis[i, :]
            x = transformation_matrices[camera_name][0, 3]
            y = transformation_matrices[camera_name][1, 3]
            z = transformation_matrices[camera_name][2, 3]
            u = orientation[0]
            v = orientation[1]
            w = orientation[2]
            ax.quiver(x, y, z, u, v, w, color = colors[i], length=1)
            
            if i == 0:
                ax.text(x, y, z, camera_name, color = 'black')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
def chessboard_projection(settings):
    mtxs = {}
    dists = {}
    transformation_matrices = {}
    for camera_name in settings.cameras:
        mtx = np.load(f'results/mtx_{camera_name}.npy')
        dist = np.load(f'results/dist_{camera_name}.npy')
        mtxs[camera_name] = mtx
        dists[camera_name] = dist
        transformation_matrix = np.load(f'results/extrinsic_{camera_name}.npy')
        transformation_matrices[camera_name] = transformation_matrix
            
    i = 1
    fig = plt.figure(figsize=(16, 16))

    for camera_name in settings.cameras:
        images = os.listdir(image_path)
        if camera_name == 'wide':
            continue
        while True:
            random_index = random.randint(0, len(images) - 1)
            frame = cv2.imread(f'{image_path}/{images[random_index]}')
            frames = slicing_frame3_1(frame)
            wide_img = frames[settings.cameras.index('wide')]
            cam_img = frames[settings.cameras.index(camera_name)]
            
            gray_wide = cv2.cvtColor(wide_img, cv2.COLOR_BGR2GRAY)
            gray_cam = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)
            ret_wide, corners_wide, objp = get_points_single_frame(gray_wide, settings, 'extrinsic')
            ret_cam, corners_cam, objp = get_points_single_frame(gray_cam, settings, 'extrinsic')
            
            if ret_wide == True and ret_cam == True:
                ret_wide,rvecs_wide, tvecs_wide = cv2.solvePnP(objp, corners_wide, mtxs[camera_name], dists[camera_name])
                ret_cam,rvecs_cam, tvecs_cam = cv2.solvePnP(objp, corners_cam, mtxs[camera_name], dists[camera_name])
                if ret_wide == True and ret_cam == True:
                    ax = fig.add_subplot(2, 2, i, projection='3d')
                    transformation_matrix_wide = np.eye(4)
                    R_cam, _ = cv2.Rodrigues(rvecs_cam)
                    R_wide, _ = cv2.Rodrigues(rvecs_wide)
                    transformation_matrix_wide[:3, :3] = R_wide
                    transformation_matrix_wide[:3, 3] = tvecs_wide.reshape(3)
                    transformation_matrix_cam = np.eye(4)
                    transformation_matrix_cam[:3, :3] = R_cam
                    transformation_matrix_cam[:3, 3] = tvecs_cam.reshape(3)
                    objp_homogeneous = np.concatenate((objp, np.ones((objp.shape[0], 1))), axis=1)
                    coord_thru_cam = transformation_matrix_cam @ objp_homogeneous.T
                    coord_thru_wide = transformation_matrix_wide @ objp_homogeneous.T
                    projected_coord = transformation_matrices[camera_name] @ coord_thru_cam
                    RMS_error = np.sqrt(np.mean((projected_coord[:3, :] - coord_thru_wide[:3, :])**2))
                    space_between_points_cam = np.sqrt(np.mean(((coord_thru_cam[:3, :].T - coord_thru_cam[:3, 0]) - objp) ** 2))
                    space_between_points_wide = np.sqrt(np.mean(((coord_thru_wide[:3, :].T - coord_thru_wide[:3, 0]) - objp) ** 2))
                    print(f'{camera_name} has RMS projection error to wide: {RMS_error}, RMS projection error to ground truth: {space_between_points_cam}. RMS error between wide and ground truth: {space_between_points_wide}')
                    
                    ax.scatter(coord_thru_cam[0, :], coord_thru_cam[1, :], coord_thru_cam[2, :], color='red')
                    ax.scatter(projected_coord[0, :], projected_coord[1, :], projected_coord[2, :], color='blue')
                    ax.set_title(f'{camera_name}')
                    break
        i += 1
    
def main():
    fig = plt.figure(figsize=(15, 10))
    visualize(settings, fig, 45, 0, 0, i=1)
    visualize(settings, fig, elev=45, azim=45, roll=0, i=2)
    visualize(settings, fig, elev=-90, azim=90, roll=0, i=3)
    plt.savefig('results/visualize_ext.png')
    chessboard_projection(settings)
    plt.savefig('results/chessboard_projection.png')
    
if __name__ == '__main__':
    main()