import matplotlib.pyplot as plt
import numpy as np
import json
import os
from matplotlib.animation import FuncAnimation

class ResultAnalysis:
    def __init__(self, file_path):
        self.data = {}
        
        try:
            with open('setting.json', 'r') as f:
                self.settings = json.load(f)
        except:
            raise Exception(f"Error loading settings from setting.json")
        
        self.load_data(file_path)
        self.remove_results()

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
            
    def remove_results(self):
        os.makedirs('result/', exist_ok=True)
        files = os.listdir('result/')
        for file in files:
            os.remove(f'result/{file}')

    def extract_position(self, camera_name1, camera_name2 = None):
        camera1_coordinates = []
        camera2_coordinates = []
        for frame_data in self.data:
            if len(frame_data[camera_name1]) > 0 and (camera_name2 is None or len(frame_data[camera_name2]) > 0):
                camera1_coordinates.append([frame_data[camera_name1][0]['x'], frame_data[camera_name1][0]['y'], frame_data[camera_name1][0]['z']])
                if camera_name2 is not None:
                    camera2_coordinates.append([frame_data[camera_name2][0]['x'], frame_data[camera_name2][0]['y'], frame_data[camera_name2][0]['z']])
                
        camera1_coordinates = np.array(camera1_coordinates)
        camera2_coordinates = np.array(camera2_coordinates)
        return camera1_coordinates, camera2_coordinates
            
    def calculate_inter_camera_difference(self, camera_name):
        camera_coordinates, center_camera_coordinates = self.extract_position(camera_name, self.settings['center_camera'])
                
        difference = camera_coordinates - center_camera_coordinates
        difference_mean = np.mean(difference, axis=0)
        difference_std = np.std(difference, axis=0)
        print(f'Difference between {camera_name} and {self.settings["center_camera"]} mean: {difference_mean}')
        print(f'Difference between {camera_name} and {self.settings["center_camera"]} std: {difference_std}')
        os.makedirs('result/', exist_ok=True)
        with open(f'result/difference.log', 'a') as f:
            f.write(f'Difference between {camera_name} and {self.settings["center_camera"]} mean: {difference_mean}\n')
            f.write(f'Difference between {camera_name} and {self.settings["center_camera"]} std: {difference_std}\n')
        
    def visualize_points(self, camera_name):
        camera_coordinates, _ = self.extract_position(camera_name)
        
        fig = plt.figure(figsize=(10, 10))
        self.draw_points(fig, 1, camera_name, camera_coordinates, 0)
        self.draw_points(fig, 2, camera_name, camera_coordinates, 1)
        self.draw_points(fig, 3, camera_name, camera_coordinates, 2)
        plt.savefig(f'result/visualize_points_{camera_name}.png')
        plt.close(fig)
        
    def visualize_points_multi_cameras(self):
        fig = plt.figure(figsize=(10, 10))
        camera_coordinates = {f'{camera_name}': [] for camera_name in self.settings['cameras']}

        for camera_name in self.settings['cameras']:
            camera_coordinates[camera_name], _ = self.extract_position(camera_name)
        
        self.draw_points_multi_cameras(fig, 1, camera_coordinates, 0)
        self.draw_points_multi_cameras(fig, 2, camera_coordinates, 1)
        self.draw_points_multi_cameras(fig, 3, camera_coordinates, 2)
        plt.savefig(f'result/visualize_points_multi_cameras.png')
        plt.close(fig)
    
    def draw_points(self, fig, i, camera_name, points, dimension_erase):
        dimenstion_kept = [0, 1, 2]
        dimenstion_kept.remove(dimension_erase)
        xyz_array = ['X', 'Y', 'Z']
        
        ax = fig.add_subplot(1, 3, i)
        ax.set_box_aspect(1)
        
        ax.set_xlabel(xyz_array[dimenstion_kept[0]])
        ax.set_ylabel(xyz_array[dimenstion_kept[1]])
        ax.scatter(points[:, dimenstion_kept[0]], points[:, dimenstion_kept[1]], color='red', alpha=0.5)
        ax.set_title(f'{camera_name}')
        
    
    def draw_points_multi_cameras(self, fig, i, points, dimension_erase):
        dimenstion_kept = [0, 1, 2]
        dimenstion_kept.remove(dimension_erase)
        xyz_array = ['X', 'Y', 'Z']
        colors = ['red', 'green', 'blue', 'yellow', 'purple']
        
        ax = fig.add_subplot(1, 3, i)
        ax.set_box_aspect(1)

        ax.set_xlabel(xyz_array[dimenstion_kept[0]])
        ax.set_ylabel(xyz_array[dimenstion_kept[1]])
        for camera_name in self.settings['cameras']:
            ax.scatter(points[camera_name][:, dimenstion_kept[0]], points[camera_name][:, dimenstion_kept[1]], color=colors[self.settings['cameras'].index(camera_name)], alpha=0.5)
        ax.set_title(f'Multi Cameras')
        ax.legend(self.settings['cameras'])

    def animate_results(self):
        print("Generating animation...")
        # Collect all points to determine axis limits
        all_points = []
        for camera_name in self.settings['cameras']:
            pts, _ = self.extract_position(camera_name)
            if len(pts) > 0:
                all_points.append(pts)
        
        if not all_points:
            print("No points to animate.")
            return

        all_points = np.concatenate(all_points, axis=0)
        min_coord = np.min(all_points, axis=0)
        max_coord = np.max(all_points, axis=0)
        padding = (max_coord - min_coord) * 0.1
        min_lim = min_coord - padding
        max_lim = max_coord + padding

        fig = plt.figure(figsize=(5, 5))
        
        # XY plot only (erase_dim = 2)
        erase_dim = 2
        keep_dims = [0, 1, 2]
        keep_dims.remove(erase_dim)
        x_idx, y_idx = keep_dims[0], keep_dims[1]
        xyz_labels = ['X', 'Y', 'Z']
        
        ax = fig.add_subplot(1, 1, 1)
        ax.set_box_aspect(1)
        ax.set_xlabel(xyz_labels[x_idx])
        ax.set_ylabel(xyz_labels[y_idx])
        ax.set_xlim(min_lim[x_idx], max_lim[x_idx])
        ax.set_ylim(min_lim[y_idx], max_lim[y_idx])
        
        colors = ['red', 'green', 'blue', 'yellow', 'purple']
        scatters = {} 
        history = {camera_name: [] for camera_name in self.settings['cameras']}
        
        # Initialize scatters
        for cam_idx, camera_name in enumerate(self.settings['cameras']):
            base_color = colors[cam_idx % len(colors)]
            # We will update facecolors/edgecolors dynamically
            scat = ax.scatter([], [], color=base_color, label=camera_name, s=20)
            scatters[camera_name] = scat
            
        ax.legend()

        def update(frame_idx):
            frame_data = self.data[frame_idx]
            artists = []
            
            for cam_idx, camera_name in enumerate(self.settings['cameras']):
                if camera_name in frame_data and len(frame_data[camera_name]) > 0:
                    pos = frame_data[camera_name][0]
                    point = [pos['x'], pos['y'], pos['z']]
                    
                    history[camera_name].append(point)
                    if len(history[camera_name]) > 100:
                        history[camera_name].pop(0)
                    
                    # Get points for current view
                    pts = np.array(history[camera_name])
                    view_pts = pts[:, [x_idx, y_idx]]
                    
                    # Calculate alpha values
                    n_points = len(view_pts)
                    alphas = np.linspace(0.01, 1, n_points)
                    
                    # Create color array
                    base_color = colors[cam_idx % len(colors)]
                    # Convert base color to RGB
                    from matplotlib.colors import to_rgb
                    rgb = to_rgb(base_color)
                    rgba_colors = np.zeros((n_points, 4))
                    rgba_colors[:, 0:3] = rgb
                    rgba_colors[:, 3] = alphas
                    
                    scatters[camera_name].set_offsets(view_pts)
                    scatters[camera_name].set_color(rgba_colors)
                    artists.append(scatters[camera_name])
                else:
                    # No update for this camera if no data, but keep history? 
                    # Or clear if lost? Assuming we keep showing history until it expires or we could just not update.
                    # But since we redraw history every frame, if we don't update history, the old points stay static.
                    # If we want them to fade out even if no new data comes, we would need to manage time differently.
                    # For now, let's assume if no data, we just show what we have or empty. 
                    # Typically, if tracking is lost, we might want to see last knowns or nothing.
                    # Let's just keep showing current history without adding new points.
                    
                    if history[camera_name]:
                        pts = np.array(history[camera_name])
                        view_pts = pts[:, [x_idx, y_idx]]
                        
                        n_points = len(view_pts)
                        # Optional: fade out history if we wanted to implement "time passing" logic without new points
                        alphas = np.linspace(0.01, 1, n_points)
                        
                        base_color = colors[cam_idx % len(colors)]
                        from matplotlib.colors import to_rgb
                        rgb = to_rgb(base_color)
                        rgba_colors = np.zeros((n_points, 4))
                        rgba_colors[:, 0:3] = rgb
                        rgba_colors[:, 3] = alphas
                        
                        scatters[camera_name].set_offsets(view_pts)
                        scatters[camera_name].set_color(rgba_colors)
                        artists.append(scatters[camera_name])
                    else:
                        scatters[camera_name].set_offsets(np.empty((0, 2)))
                        artists.append(scatters[camera_name])

            return artists

        ani = FuncAnimation(fig, update, frames=len(self.data), interval=50, blit=True)
        save_path = 'result/animation.mp4'
        # Use ffmpeg writer
        ani.save(save_path, writer='ffmpeg', fps=20)
        plt.close(fig)
        print(f"Animation saved to {save_path}")

def main():
    result_analysis = ResultAnalysis('output/20251116_160147.json')
    result_analysis.calculate_inter_camera_difference('cam0')
    result_analysis.calculate_inter_camera_difference('cam1')
    result_analysis.calculate_inter_camera_difference('cam2')
    result_analysis.visualize_points('cam0')
    result_analysis.visualize_points('cam1')
    result_analysis.visualize_points('cam2')
    result_analysis.visualize_points_multi_cameras()
    result_analysis.animate_results()
    
if __name__ == "__main__":
    main()