import numpy as np
from settings_loader import settings
import matplotlib.pyplot as plt
import pickle
import os


def visualize():
    dpi = 100
    fig = plt.figure(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title('3D Point Cloud')
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    points = np.load('output/points.npy')
    rets = np.load('output/rets.npy')
    
    for camera_name in settings.cameras:
        camera_index = settings.cameras.index(camera_name)
        camera_rets = rets[:, camera_index]
        points_camera = points[camera_rets, camera_index, :]
        ax.scatter(points_camera[:, 0], points_camera[:, 1], points_camera[:, 2], 
                    color=colors[camera_index],
                    alpha=0.3, marker='.')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal aspect ratio and center the plot
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio for all axes
    
    # Get the data statistics to center the plot
    all_points = points.reshape(-1, 3)
    x_std = np.std(all_points[:, 0])
    y_std = np.std(all_points[:, 1])
    z_std = np.std(all_points[:, 2])
    
    # Use the maximum standard deviation for all axes to keep consistent scale
    max_std = max(x_std, y_std, z_std)
    
    # Center the plot using mean values
    x_center = np.mean(all_points[:, 0])
    y_center = np.mean(all_points[:, 1])
    z_center = np.mean(all_points[:, 2])
    
    # Set equal limits for all axes using standard deviation (typically 3*std covers ~99.7% of data)
    multiplier = 3  # You can adjust this multiplier (2 for ~95%, 3 for ~99.7%)
    ax.set_xlim(x_center - multiplier * max_std, x_center + multiplier * max_std)
    ax.set_ylim(y_center - multiplier * max_std, y_center + multiplier * max_std)
    ax.set_zlim(z_center - multiplier * max_std, z_center + multiplier * max_std)

    legend = ax.legend(settings.cameras, title='Cameras')
    legend.set_bbox_to_anchor((1.0, 1.0))
    
    # Show the plot in non-blocking mode
    plt.show()
    
def calculate_difference():
    points = np.load('output/points.npy')
    rets = np.load('output/rets.npy')
    for camera_name in settings.cameras:
        if camera_name == 'wide':
            continue
        camera_index = settings.cameras.index(camera_name)
        wide_index = settings.cameras.index('wide')
        camera_rets = rets[:, camera_index]
        wide_rets = rets[:, wide_index]
        shared_rets = camera_rets & wide_rets
        points_camera = points[shared_rets, camera_index, :]
        wide_points = points[shared_rets, wide_index, :]
        difference = points_camera - wide_points
        print(f'{camera_name} has difference: {np.mean(difference, axis=0)}')
        with open('output/error.json', 'a') as f:
            f.write(f'{camera_name}_mean: {np.sqrt(np.mean(difference*difference, axis=0))}\n')
            f.write(f'{camera_name}_std: {np.std(difference, axis=0)}\n')
        
def visualize_single_frame(frame_points, frame_index, frame):
    frame_points = np.array(frame_points)
    frame_index = np.array(frame_index)
    dpi = 100
    fig = plt.figure(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_title('3D Point Cloud')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1,1,1])
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    for camera_name in settings.cameras:
        camera_index = settings.cameras.index(camera_name)
        points_camera = frame_points[camera_index, :]
        ax.scatter(points_camera[0], points_camera[1], points_camera[2], 
                    color=colors[camera_index],
                    alpha=1, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    legend = ax.legend(settings.cameras, title='Cameras')
    legend.set_bbox_to_anchor((1.0, 1.0))
    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(frame, cmap='gray')
    ax.set_title('Frame')
    ax = fig.add_subplot(2, 2, 3)
    import pandas as pd
    table_data = []
    for camera_name in settings.cameras:
        camera_index = settings.cameras.index(camera_name)
        coords = frame_points[camera_index, :]
        table_data.append({
            'Camera': camera_name,
            'X': coords[0],
            'Y': coords[1],
            'Z': coords[2]
        })
    df = pd.DataFrame(table_data)
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    os.makedirs('output/shared', exist_ok=True)
    plt.savefig(f'output/shared/single_frame_{frame_index}.png')
    plt.close(fig)
    
        
def main():
    visualize()
    calculate_difference()
    
if __name__ == '__main__':
    main()