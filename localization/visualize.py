import numpy as np
from settings_loader import settings
import matplotlib.pyplot as plt
import matplotlib
import signal
import sys
import pickle

# Set matplotlib to use a non-blocking backend
matplotlib.use('TkAgg')

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\nExiting visualization...')
    plt.close('all')
    sys.exit(0)

def visualize(points):
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Enable interactive mode
    plt.ion()
    
    dpi = 100
    fig = plt.figure(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title('3D Point Cloud - Press Ctrl+C to exit')
    draw_detection_graph(ax, points)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Show the plot in non-blocking mode
    plt.show(block=False)
    
    # Keep the window open and responsive
    try:
        while plt.get_fignums():
            plt.pause(0.1)
    except KeyboardInterrupt:
        print('\nExiting visualization...')
    finally:
        plt.close('all')
    

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
    points = pickle.load(open('output/points.pkl', 'rb'))
    visualize(points)
    
if __name__ == '__main__':
    main()