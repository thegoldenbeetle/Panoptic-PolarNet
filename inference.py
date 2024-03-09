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

from network.BEV_Unet import BEV_Unet
from network.ptBEV import ptBEVnet
from dataloader.dataset import collate_fn_BEV,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset,collate_fn_BEV_test, process_one_frame
from network.instance_post_processing import get_panoptic_segmentation
from utils.eval_pq import PanopticEval
from utils.configs import merge_configs
from dataloader.process_panoptic import PanopticLabelGenerator

#ignore weird np warning
import warnings
warnings.filterwarnings("ignore")

def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)

def SemKITTI2train_single(label):
    return label - 1 # uint8 trick


def main(args, data_tuple):
    
    data_path = args['dataset']['path']
    test_batch_size = args['model']['test_batch_size']
    pretrained_model = args['model']['pretrained_model']
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
    if os.path.exists(pretrained_model):
        my_model.load_state_dict(torch.load(pretrained_model))
    pytorch_total_params = sum(p.numel() for p in my_model.parameters())
    print('params: ',pytorch_total_params)
    my_model.to(pytorch_device)
    my_model.eval()

    run_inference_one_frame(my_model, data_tuple)

    # prepare dataset
    # test_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'test', return_ref = True, instance_pkl_path=args['dataset']['instance_pkl_path'])
    # if args['model']['polar']:
    #     test_dataset=spherical_dataset(test_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0, return_test= True)
    # else:
    #     test_dataset=voxel_dataset(test_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0, return_test= True)

    # print("!!!!!! Len ", len(test_dataset))


    # test_dataset_loader = torch.utils.data.DataLoader(dataset = test_dataset,
    #                                                 batch_size = test_batch_size,
    #                                                 collate_fn = collate_fn_BEV_test,
    #                                                 shuffle = False,
    #                                                 num_workers = 4)
    
    # # test
    # print('*'*80)
    # print('Generate predictions for test split')
    # print('*'*80)
    # pbar = tqdm(total=len(test_dataset_loader))
    # with torch.no_grad():
    #     for i_iter_test,(test_vox_fea,_,_,_,test_grid,_,_,test_pt_fea,test_index) in enumerate(test_dataset_loader):
    #         # predict
    #         test_vox_fea_ten = test_vox_fea.to(pytorch_device)
    #         test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in test_pt_fea]
    #         test_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in test_grid]

    #         if visibility:
    #             predict_labels,center,offset = my_model(test_pt_fea_ten,test_grid_ten,test_vox_fea_ten)
    #         else:
    #             predict_labels,center,offset = my_model(test_pt_fea_ten,test_grid_ten)
    #         # write to label file
    #         for count,i_test_grid in enumerate(test_grid):
    #             # get foreground_mask
    #             for_mask = torch.zeros(1,grid_size[0],grid_size[1],grid_size[2],dtype=torch.bool).to(pytorch_device)
    #             for_mask[0,test_grid[count][:,0],test_grid[count][:,1],test_grid[count][:,2]] = True
    #             # post processing
    #             panoptic_labels,center_points = get_panoptic_segmentation(torch.unsqueeze(predict_labels[count], 0),torch.unsqueeze(center[count], 0),torch.unsqueeze(offset[count], 0),test_pt_dataset.thing_list,\
    #                                                                                       threshold=args['model']['post_proc']['threshold'], nms_kernel=args['model']['post_proc']['nms_kernel'],\
    #                                                                                       top_k=args['model']['post_proc']['top_k'], polar=circular_padding,foreground_mask=for_mask)
    #             panoptic_labels = panoptic_labels.cpu().detach().numpy().astype(np.uint32)
    #             panoptic = panoptic_labels[0,test_grid[count][:,0],test_grid[count][:,1],test_grid[count][:,2]]
    #             save_dir = test_pt_dataset.im_idx[test_index[count]]
    #             _,dir2 = save_dir.split('/sequences/',1)
    #             new_save_dir = output_path + '/sequences/' +dir2.replace('velodyne','predictions')[:-3]+'label'
    #             if not os.path.exists(os.path.dirname(new_save_dir)):
    #                 try:
    #                     os.makedirs(os.path.dirname(new_save_dir))
    #                 except OSError as exc:
    #                     if exc.errno != errno.EEXIST:
    #                         raise
    #             panoptic.tofile(new_save_dir)
    #         del test_pt_fea_ten,test_grid_ten,test_pt_fea,predict_labels,center,offset
    #         pbar.update(1)
    # pbar.close()
    # print('Predicted test labels are saved in %s. Need to be shifted to original label format before submitting to the Competition website.' % output_path)
    # print('Remapping script can be found in semantic-kitti-api.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--configs', default='configs/SemanticKITTI_model/Panoptic-PolarNet.yaml')    
    args = parser.parse_args()
    
    with open(args.configs, 'r') as s:
        new_args = yaml.safe_load(s)    
    

    grid_size = new_args['dataset']['grid_size']
    grid_size = np.asarray(grid_size)
    panoptic_proc = PanopticLabelGenerator(grid_size, sigma = new_args['dataset']['gt_generator']['sigma'], polar = True) 
    data_tuple = process_one_frame("/data/sequences/21/velodyne/002695.bin", grid_size, panoptic_proc)

    main(new_args, data_tuple)