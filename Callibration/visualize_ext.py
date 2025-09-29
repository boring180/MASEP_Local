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
from get_points import get_points_single_frame
from settings_loader import settings

sys.path.append(os.path.dirname(os.path.abspath('.')))
from utils.frame_slicing import slicing_frame3_1, slicing_frame3_2
image_path = '../photos/multi_camera'

homo_point_array = np.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]])
homo_point_array = homo_point_array * settings.pattern_square_size_external

def visualize_camera_location(settings, fig, elev, azim, roll, i):
    transformation_matrices = {}
    for camera_name in settings.cameras:
        transformation_matrix = pickle.load(open(f'results/extrinsic_{camera_name}.pkl', 'rb'))
        transformation_matrices[camera_name] = transformation_matrix
            
    ax = fig.add_subplot(1, 3, i, projection='3d', elev=elev, azim=azim, roll=roll)
    ax.set_box_aspect([1,1,1])
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
        
def visualize_objp(fig, i, coord_thru_cam, projected_coord):
    ax = fig.add_subplot(1, 2, i, projection='3d')
    X_max = max(np.max(coord_thru_cam[0, :]), np.max(projected_coord[0, :]))
    X_min = min(np.min(coord_thru_cam[0, :]), np.min(projected_coord[0, :]))
    Y_max = max(np.max(coord_thru_cam[1, :]), np.max(projected_coord[1, :]))
    Y_min = min(np.min(coord_thru_cam[1, :]), np.min(projected_coord[1, :]))
    Z_max = max(np.max(coord_thru_cam[2, :]), np.max(projected_coord[2, :]))
    Z_min = min(np.min(coord_thru_cam[2, :]), np.min(projected_coord[2, :]))
    X_range = X_max - X_min
    Y_range = Y_max - Y_min
    Z_range = Z_max - Z_min
    max_range = max(X_range, Y_range, Z_range)
    ax.set_xlim(X_min - max_range * 0.1, X_min + max_range * 1.1)
    ax.set_ylim(Y_min - max_range * 0.1, Y_min + max_range * 1.1)
    ax.set_zlim(Z_min - max_range * 0.1, Z_min + max_range * 1.1)
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(coord_thru_cam[0, :], coord_thru_cam[1, :], coord_thru_cam[2, :], color='red')
    ax.scatter(projected_coord[0, :], projected_coord[1, :], projected_coord[2, :], color='blue')
    ax.set_title(f'{settings.cameras[i]}')
        
def chessboard_projection(settings):
    mtxs = {}
    dists = {}
    transformation_matrices = {}
    for camera_name in settings.cameras:
        mtx = pickle.load(open(f'results/mtx_{camera_name}.pkl', 'rb'))
        dist = pickle.load(open(f'results/dist_{camera_name}.pkl', 'rb'))
        mtxs[camera_name] = mtx
        dists[camera_name] = dist
        transformation_matrix = pickle.load(open(f'results/extrinsic_{camera_name}.pkl', 'rb'))
        transformation_matrices[camera_name] = transformation_matrix
            
    i = 1
    fig = plt.figure(figsize=(16, 16))

    for camera_name in settings.cameras:
        images = os.listdir(image_path)
        center_camera = settings.center_camera
        if camera_name == center_camera:
            continue
        number_of_success_points = 0
        NUMBER_OF_SUCCESS_POINTS_GOAL = 1000
        
        worst_case_error = 0
        worst_case_frame = None
        worst_case_coord_thru_center = None
        worst_case_projected_coord = None
        
        best_case_error = np.inf
        best_case_frame = None
        best_case_coord_thru_center = None
        best_case_projected_coord = None
        
        while True:
            random_index = random.randint(0, len(images) - 1)
            frame = cv2.imread(f'{image_path}/{images[random_index]}')
            frames = slicing_frame3_1(frame)
            center_img = frames[settings.cameras.index(center_camera)]
            cam_img = frames[settings.cameras.index(camera_name)]
            
            gray_center = cv2.cvtColor(center_img, cv2.COLOR_BGR2GRAY)
            gray_cam = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)
            ret_center, imgp_center, objp_center = get_points_single_frame(gray_center, settings, 'extrinsic')
            ret_cam, imgp_cam, objp_cam = get_points_single_frame(gray_cam, settings, 'extrinsic')
            
            error_array = []
            
            if ret_center == True and ret_cam == True:
                ret_center,rvecs_center, tvecs_center = cv2.solvePnP(objp_center, imgp_center, mtxs[camera_name], dists[camera_name])
                ret_cam,rvecs_cam, tvecs_cam = cv2.solvePnP(objp_cam, imgp_cam, mtxs[center_camera], dists[center_camera])
                if ret_center == True and ret_cam == True:
                    R_cam, _ = cv2.Rodrigues(rvecs_cam)
                    R_center, _ = cv2.Rodrigues(rvecs_center)
                    transformation_matrix_center = np.eye(4)
                    transformation_matrix_center[:3, :3] = R_center
                    transformation_matrix_center[:3, 3] = tvecs_center.reshape(3)
                    transformation_matrix_cam = np.eye(4)
                    transformation_matrix_cam[:3, :3] = R_cam
                    transformation_matrix_cam[:3, 3] = tvecs_cam.reshape(3)

                    coord_thru_cam = transformation_matrix_cam @ homo_point_array.T
                    coord_thru_center = transformation_matrix_center @ homo_point_array.T
                    projected_coord = transformation_matrices[camera_name] @ coord_thru_cam
                    
                    error = projected_coord[:3, :] - coord_thru_center[:3, :]
                    error_array.append(error)
                    frame_mean_error = np.mean(error, axis=1)
                    RMS_error = np.sqrt(np.sum(frame_mean_error * frame_mean_error))
                    
                    if RMS_error > worst_case_error:
                        worst_case_error = RMS_error
                        worst_case_frame = frame
                        worst_case_coord_thru_center = coord_thru_center
                        worst_case_projected_coord = projected_coord
                        
                    if RMS_error < best_case_error:
                        best_case_error = RMS_error
                        best_case_frame = frame
                        best_case_coord_thru_center = coord_thru_center
                        best_case_projected_coord = projected_coord
                        
                    number_of_success_points += imgp_cam.shape[0]
                    if number_of_success_points >= NUMBER_OF_SUCCESS_POINTS_GOAL:
                        break
                    
        mean_error = np.mean(error_array, axis=0)
        std_error = np.std(error_array, axis=0)
                    
        print(f'{camera_name} has RMS projection error to center: {mean_error}, with std: {std_error}, with worst case error: {worst_case_error}, with best case error: {best_case_error}')
        with open(f'results/extrinsic_calibration.log', 'a') as f:
            f.write(f'{camera_name} has RMS projection error to center: {mean_error}, with std: {std_error}, with worst case error: {worst_case_error}, with best case error: {best_case_error}\n')
            
        visualize_objp(fig, i, best_case_coord_thru_center, best_case_projected_coord)
        
        break
        i += 1
    
def main():
    fig = plt.figure(figsize=(15, 10))
    visualize_camera_location(settings, fig, 45, 0, 0, i=1)
    visualize_camera_location(settings, fig, elev=45, azim=45, roll=0, i=2)
    visualize_camera_location(settings, fig, elev=-90, azim=90, roll=0, i=3)
    plt.savefig('results/visualize_ext.png')
    chessboard_projection(settings)
    plt.savefig('results/chessboard_projection.png')
    
if __name__ == '__main__':
    main()