import time
import open3d as o3d
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import argparse


def animate_point_clouds(point_clouds):
    
    # Initialize point cloud geometry
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_clouds[0][:, :3])
    point_cloud.colors = o3d.utility.Vector3dVector(np.asarray([[0, 0, reflectance] for reflectance in point_clouds[0][:, 3]]))

    # Create Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Set background color to black
    vis.get_render_option().background_color = np.array([0, 0, 0])
    vis.add_geometry(point_cloud)

    if len(point_clouds) == 1:
        vis.run()

    else:
        frame_index = 0
        last_update_time = time.time()
        update_interval = 0.25  # Time in seconds between frame updates
        while True:
            current_time = time.time()
            if current_time - last_update_time > update_interval:
                
                # Update point cloud with new data
                point_cloud.points = o3d.utility.Vector3dVector(point_clouds[frame_index][:, :3])
                point_cloud.colors = o3d.utility.Vector3dVector(np.asarray([[0, 0, reflectance] for reflectance in point_clouds[frame_index][:, 3]]))
                vis.update_geometry(point_cloud)
                
                # Move to the next frame
                frame_index = (frame_index + 1) % len(point_clouds)
                last_update_time = current_time
            vis.poll_events()
            vis.update_renderer()
            if not vis.poll_events():
                break

    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-p', '--path', default="/data_kitti/08/velodyne/001114.bin")    
    parser.add_argument('-l', '--label', default="/data_kitti/08/velodyne/001114.bin")    
    args = parser.parse_args()

    data_path = args.path

    if os.path.isfile(data_path):
        files = [data_path]
    else:
        files = sorted([join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))])

    pointclouds = [np.fromfile(x, dtype=np.float32).reshape((-1, 4)) for x in files]
    animate_point_clouds(pointclouds)
