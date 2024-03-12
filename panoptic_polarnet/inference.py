#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
import sys
import yaml
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import errno

from panoptic_polarnet.network.BEV_Unet import BEV_Unet
from panoptic_polarnet.network.ptBEV import ptBEVnet
from panoptic_polarnet.dataloader.dataset import collate_fn_BEV,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset,collate_fn_BEV_test
from panoptic_polarnet.dataloader.data_inference import process_one_frame
from panoptic_polarnet.network.instance_post_processing import get_panoptic_segmentation
from panoptic_polarnet.utils.eval_pq import PanopticEval
from panoptic_polarnet.utils.configs import merge_configs
from panoptic_polarnet.dataloader.process_panoptic import PanopticLabelGenerator

#ignore weird np warning
import warnings
warnings.filterwarnings("ignore")

######## Onnx scatter
import torch
import torch.nn as nn
from torch.onnx import symbolic_helper
from torch_scatter import scatter_max

class ScatterMax(nn.Module):

    def forward(self, src: torch.Tensor, index: torch.Tensor):
        # src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
        # index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        out, argmax = scatter_max(src, index, dim=-1, out=src)
        return out, argmax


@symbolic_helper.parse_args("v", "v", "i", "v", "i", "i")
def symbolic_scatter_max(g, src, index, dim=-1, out=None, dim_size=None, fill_value=None):
    return (index ,g.op("ScatterElements", out, index, src, axis_i=dim, reduction_s="max"))



def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)

def SemKITTI2train_single(label):
    return label - 1 # uint8 trick


def run_inference_one_frame(my_model, input, grid_size, thing_list):
    
    with torch.no_grad():

        test_vox_fea, _, _, _, test_grid, _, _, test_pt_fea = input

        pytorch_device = torch.device('cuda:0')    
        
        test_vox_fea = torch.from_numpy(np.expand_dims(test_vox_fea, axis=0).astype(np.float32))
        test_grid = np.expand_dims(test_grid, axis=0)
        test_pt_fea = np.expand_dims(test_pt_fea, axis=0)

        # predict
        test_vox_fea_ten = test_vox_fea.to(pytorch_device)
        test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in test_pt_fea]
        test_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in test_grid]


        # model
        predict_labels, center, offset = my_model(test_pt_fea_ten, test_grid_ten, test_vox_fea_ten)

        # # convert to onnx
        # torch.onnx.register_custom_op_symbolic("torch_scatter::scatter_max", symbolic_scatter_max, 18)


        # torch.onnx.export(my_model,               # model being run
        #                   (test_pt_fea_ten, test_grid_ten, test_vox_fea_ten),                         # model input (or a tuple for multiple inputs)
        #                   "test.onnx",   # where to save the model (can be a file or file-like object)
        #                   export_params=True,        # store the trained parameter weights inside the model file
        #                   opset_version=18,          # the ONNX version to export the model to
        #                   do_constant_folding=True,  # whether to execute constant folding for optimization
        #                   input_names = ['input'],   # the model's input names
        #                   output_names = ['output'], # the model's output names
        #                 )


        # get labels
        count = 0

        # get foreground_mask
        for_mask = torch.zeros(1, grid_size[0], grid_size[1], grid_size[2], dtype=torch.bool).to(pytorch_device)
        for_mask[0, test_grid[count][:, 0], test_grid[count][:, 1], test_grid[count][:, 2]] = True

        # post processing
        panoptic_labels, center_points = get_panoptic_segmentation(
            torch.unsqueeze(predict_labels[0], 0),
            torch.unsqueeze(center[0], 0),
            torch.unsqueeze(offset[0], 0),
            thing_list,
            threshold=0.1 ,#args['model']['post_proc']['threshold'], 
            nms_kernel=5, #args['model']['post_proc']['nms_kernel'],
            top_k=100, #args['model']['post_proc']['top_k'],
            polar=True,
            foreground_mask=for_mask
        )
        panoptic_labels = panoptic_labels.cpu().detach().numpy().astype(np.uint32)

        panoptic = panoptic_labels[0, test_grid[count][:, 0], test_grid[count][:, 1], test_grid[count][:, 2]]

        panoptic.tofile("test.label")



def main(args, data_tuple, thing_list):
    
    data_path = args['dataset']['path']
    test_batch_size = args['model']['test_batch_size']
    # pretrained_model = args['model']['pretrained_model']
    pretrained_model = "pretrained_weight/Panoptic_SemKITTI_PolarNet.pt"
    output_path = args['dataset']['output_path']
    compression_model = args['dataset']['grid_size'][2]
    grid_size = args['dataset']['grid_size']
    visibility = args['model']['visibility']
    pytorch_device = torch.device('cuda:0')
    if args['model']['polar']:
        fea_dim = 9
        circular_padding = True
    else:
        fea_dim = 7
        circular_padding = False

    # !!! prepare miou fun
    unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str=[SemKITTI_label_name[x] for x in unique_label+1]

    # prepare model
    my_BEV_model=BEV_Unet(n_class=len(unique_label), n_height = compression_model, input_batch_norm = True, dropout = 0.5, circular_padding = circular_padding, use_vis_fea=visibility)
    my_model = ptBEVnet(my_BEV_model, pt_model = 'pointnet', grid_size =  grid_size, fea_dim = fea_dim, max_pt_per_encode = 256,
                            out_pt_fea_dim = 512, kernal_size = 1, pt_selection = 'random', fea_compre = compression_model)
    print(os.path.exists(pretrained_model))
    if os.path.exists(pretrained_model):
        my_model.load_state_dict(torch.load(pretrained_model))
    pytorch_total_params = sum(p.numel() for p in my_model.parameters())
    print('params: ',pytorch_total_params)
    my_model.to(pytorch_device)
    my_model.eval()

    print("Run inference one frame")
    run_inference_one_frame(my_model, data_tuple, grid_size, thing_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--configs', default='configs/SemanticKITTI_model/Panoptic-PolarNet.yaml')    
    args = parser.parse_args()
    
    with open(args.configs, 'r') as s:
        new_args = yaml.safe_load(s)    
    

    grid_size = new_args['dataset']['grid_size']
    grid_size = np.asarray(grid_size)
    panoptic_proc = PanopticLabelGenerator(grid_size, sigma = new_args['dataset']['gt_generator']['sigma'], polar = True) 
    data_tuple, thing_list  = process_one_frame("/data/sequences/21/velodyne/000000.bin", grid_size, panoptic_proc)

    main(new_args, data_tuple, thing_list)