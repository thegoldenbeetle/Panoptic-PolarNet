import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
import pickle
import errno
from torch.utils import data

from .process_panoptic import PanopticLabelGenerator
from .instance_augmentation import instance_augmentation


def process_one_frame(
    path_to_bin,
    grid_size,
    panoptic_proc,
    return_ref=True,
    fixed_volume_space=True,
    max_volume_space=[50, np.pi, 1.5],
    min_volume_space=[3, -np.pi, -3],
    ignore_label=0,
):

    # parameters
    with open("./configs/semantic-kitti.yaml", "r") as stream:
        semkittiyaml = yaml.safe_load(stream)
    thing_class = semkittiyaml["thing_class"]
    thing_list = [cl for cl, ignored in thing_class.items() if ignored]

    # read the bin file
    raw_data = np.fromfile(path_to_bin, dtype=np.float32).reshape((-1, 4))

    data_tuple = prepare_input_tuple(
        raw_data,
        grid_size,
        panoptic_proc,
        thing_list,
        return_ref,
        fixed_volume_space,
        max_volume_space,
        min_volume_space,
        ignore_label,
    )
    # what's inside input tuple
    # len = 8
    # 0 - -1 and 0, shape (32, 480, 360)
    # 1 - 0, shape (480, 360, 32)
    # 2 - 0.0, shape (1, 480, 360)
    # 3 - 0.0, shape (2, 480, 360)
    # 4 - different int numbers, shape (122302, 3)
    # 5 - 0, shape (122302, 1)
    # 6 - 0, shape (122302, 1)
    # 7 - float numbers,(122302, 9)

    return data_tuple, thing_list


def prepare_input_tuple(
    raw_data,
    grid_size,
    panoptic_proc,
    thing_list,
    return_ref=True,
    fixed_volume_space=True,
    max_volume_space=[50, np.pi, 1.5],
    min_volume_space=[3, -np.pi, -3],
    ignore_label=0,
):

    sem_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
    inst_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=np.uint32), axis=1)

    data_tuple = (raw_data[:, :3], sem_data.astype(np.uint8), inst_data)

    if return_ref:
        data_tuple += (raw_data[:, 3],)
    # data_tuple[0] - pointcloud array
    # data_tuple[1] - number of points * 1, zeros
    # data_tuple[2] - number of points * 1, zeros
    # data_tuple[3] - pointcloud reflectance

    print("My dataset: ", len(data_tuple))

    xyz, labels, insts, feat = data_tuple
    if len(feat.shape) == 1:
        feat = feat[..., np.newaxis]
    if len(labels.shape) == 1:
        labels = labels[..., np.newaxis]
    if len(insts.shape) == 1:
        insts = insts[..., np.newaxis]

    # convert coordinate into polar coordinates
    xyz_pol = cart2polar(xyz)

    # should be fixed for test / inference?
    if fixed_volume_space:
        max_bound = np.asarray(max_volume_space)
        min_bound = np.asarray(min_volume_space)

    # get grid index
    crop_range = max_bound - min_bound
    cur_grid_size = grid_size
    intervals = crop_range / (cur_grid_size - 1)
    if (intervals == 0).any():
        print("Zero interval!")
    grid_ind = (
        np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)
    ).astype(np.int64)
    current_grid = grid_ind[: np.size(labels)]

    # process voxel position
    voxel_position = np.zeros(grid_size, dtype=np.float32)
    dim_array = np.ones(len(grid_size) + 1, int)
    dim_array[0] = -1
    voxel_position = np.indices(grid_size) * intervals.reshape(
        dim_array
    ) + min_bound.reshape(dim_array)

    # process labels
    processed_label = np.ones(grid_size, dtype=np.uint8) * ignore_label
    label_voxel_pair = np.concatenate([current_grid, labels], axis=1)
    label_voxel_pair = label_voxel_pair[
        np.lexsort((current_grid[:, 0], current_grid[:, 1], current_grid[:, 2])), :
    ]
    processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

    # get thing points mask
    mask = np.zeros_like(labels, dtype=bool)
    for label in thing_list:
        mask[labels == label] = True

    inst_label = insts[mask].squeeze()
    unique_label = np.unique(inst_label)
    unique_label_dict = {label: idx + 1 for idx, label in enumerate(unique_label)}
    if inst_label.size > 1:
        inst_label = np.vectorize(unique_label_dict.__getitem__)(inst_label)

        # process panoptic
        processed_inst = np.ones(grid_size[:2], dtype=np.uint8) * ignore_label
        inst_voxel_pair = np.concatenate(
            [current_grid[mask[:, 0], :2], inst_label[..., np.newaxis]], axis=1
        )
        inst_voxel_pair = inst_voxel_pair[
            np.lexsort((current_grid[mask[:, 0], 0], current_grid[mask[:, 0], 1])), :
        ]
        processed_inst = nb_process_inst(np.copy(processed_inst), inst_voxel_pair)
    else:
        processed_inst = None

    center, center_points, offset = panoptic_proc(
        insts[mask],
        xyz[: np.size(labels)][mask[:, 0]],
        processed_inst,
        voxel_position[:2, :, :, 0],
        unique_label_dict,
        min_bound,
        intervals,
    )

    # prepare visiblity feature
    # find max distance index in each angle,height pair
    valid_label = np.zeros_like(processed_label, dtype=bool)
    valid_label[current_grid[:, 0], current_grid[:, 1], current_grid[:, 2]] = True
    valid_label = valid_label[::-1]
    max_distance_index = np.argmax(valid_label, axis=0)
    max_distance = max_bound[0] - intervals[0] * (max_distance_index)
    distance_feature = np.expand_dims(max_distance, axis=2) - np.transpose(
        voxel_position[0], (1, 2, 0)
    )
    distance_feature = np.transpose(distance_feature, (1, 2, 0))
    # convert to boolean feature
    distance_feature = (distance_feature > 0) * -1.0
    distance_feature[current_grid[:, 2], current_grid[:, 0], current_grid[:, 1]] = 1.0
    data_tuple = (distance_feature, processed_label, center, offset)
    # center data on each voxel for PTnet
    voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
    return_xyz = xyz_pol - voxel_centers
    return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

    if len(data_tuple) == 3:
        return_fea = return_xyz
    elif len(data_tuple) == 4:
        return_fea = np.concatenate((return_xyz, feat), axis=1)

    data_tuple += (grid_ind, labels, insts, return_fea)

    return data_tuple


@nb.jit("u1[:,:,:](u1[:,:,:],i8[:,:])", nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = (
                np.argmax(counter)
            )
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(
        counter
    )
    return processed_label


@nb.jit("u1[:,:](u1[:,:],i8[:,:])", nopython=True, cache=True, parallel=False)
def nb_process_inst(processed_inst, sorted_inst_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_inst_voxel_pair[0, 2]] = 1
    cur_sear_ind = sorted_inst_voxel_pair[0, :2]
    for i in range(1, sorted_inst_voxel_pair.shape[0]):
        cur_ind = sorted_inst_voxel_pair[i, :2]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_inst[cur_sear_ind[0], cur_sear_ind[1]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_inst_voxel_pair[i, 2]] += 1
    processed_inst[cur_sear_ind[0], cur_sear_ind[1]] = np.argmax(counter)
    return processed_inst


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)
