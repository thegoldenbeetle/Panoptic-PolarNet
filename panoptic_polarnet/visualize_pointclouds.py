import time
import open3d as o3d
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import argparse
from tqdm import tqdm
import yaml


def get_colors(labels, color_option, inst_color_lut, sem_color_lut):
    sem_labels = [label & 0xFFFF for label in labels]
    inst_labels = [label >> 16 for label in labels]

    if color_option == "i":
        inst_label_color = inst_color_lut[inst_labels]
        inst_label_color = inst_label_color.reshape((-1, 3))
        return inst_label_color
    if color_option == "s":
        sem_label_color = sem_color_lut[sem_labels]
        sem_label_color = sem_label_color.reshape((-1, 3))
        return sem_label_color


def init_labels_colors(semantic_color_dict):
    # make semantic colors
    max_sem_key = 0
    for key, data in semantic_color_dict.items():
        if key + 1 > max_sem_key:
            max_sem_key = key + 1
    sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
    for key, value in semantic_color_dict.items():
        sem_color_lut[key] = np.array(value, np.float32) / 255.0

    # make instance colors
    max_inst_id = 100000
    inst_color_lut = np.random.uniform(low=0.0, high=1.0, size=(max_inst_id, 3))
    # force zero to a gray-ish color
    inst_color_lut[0] = np.full((3), 0.1)

    return inst_color_lut, sem_color_lut


def animate_point_clouds(point_clouds, labels, semantic_color_dict, color_option):

    inst_color_lut, sem_color_lut = None, None
    if (color_option == "s" or color_option == "i") and labels and semantic_color_dict:
        inst_color_lut, sem_color_lut = init_labels_colors(semantic_color_dict)

    # Initialize point cloud geometry
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_clouds[0][:, :3])
    if color_option == "r":
        colors = [[0, 0, reflectance] for reflectance in point_clouds[0][:, 3]]
    else:
        colors = get_colors(labels[0], color_option, inst_color_lut, sem_color_lut)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Create Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Set background color to black
    # vis.get_render_option().background_color = np.array([0, 0, 0])
    vis.get_render_option().background_color = np.array([255, 255, 255])
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
                point_cloud.points = o3d.utility.Vector3dVector(
                    point_clouds[frame_index][:, :3]
                )
                if color_option == "r":
                    colors = [
                        [0, 0, reflectance]
                        for reflectance in point_clouds[frame_index][:, 3]
                    ]
                else:
                    colors = get_colors(
                        labels[frame_index], color_option, inst_color_lut, sem_color_lut
                    )
                point_cloud.colors = o3d.utility.Vector3dVector(colors)
                vis.update_geometry(point_cloud)

                # Move to the next frame
                frame_index = (frame_index + 1) % len(point_clouds)
                last_update_time = current_time
            vis.poll_events()
            vis.update_renderer()
            if not vis.poll_events():
                break

    vis.destroy_window()


def get_files(path):
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted([join(path, f) for f in listdir(path) if isfile(join(path, f))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-p", "--path", default="/data_kitti/08/velodyne/001114.bin")
    parser.add_argument("-l", "--label", default="/data_kitti/08/labels/001114.label")
    parser.add_argument("-y", "--yaml-config", default="./configs/semantic-kitti.yaml")
    parser.add_argument(
        "-c", "--color", choices=["r", "s", "i"], default="r"
    )  # reflectance semantic instance

    args = parser.parse_args()

    # init
    semantic_color_dict = None
    pointclouds = None
    labels = None

    if args.yaml_config and os.path.isfile(args.yaml_config):
        with open(args.yaml_config, "r") as stream:
            semkittiyaml = yaml.safe_load(stream)
            semantic_color_dict = semkittiyaml["color_map"]

    if args.path:
        points_files = get_files(args.path)
        pointclouds = [
            np.fromfile(x, dtype=np.float32).reshape((-1, 4)) for x in points_files
        ]

    if args.label:
        label_files = get_files(args.label)
        labels = [np.fromfile(x, dtype=np.uint32).reshape((-1, 1)) for x in label_files]

    animate_point_clouds(pointclouds, labels, semantic_color_dict, args.color)
