import numpy as np
from settings_loader import settings
import matplotlib.pyplot as plt
import pickle


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
    
    # Show the plot in non-blocking mode
    plt.show()
        
def main():
    visualize()
    
if __name__ == '__main__':
    main()