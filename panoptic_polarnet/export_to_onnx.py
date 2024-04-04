import argparse
import torch
import onnx
import os
import torch.nn as nn
from torch.onnx import symbolic_helper
from torch_scatter import scatter_max
from panoptic_polarnet.dataloader.process_panoptic import PanopticLabelGenerator
import numpy as np
import yaml
from panoptic_polarnet.dataloader.dataset import SemKITTI_label_name
from panoptic_polarnet.dataloader.data_inference import process_one_frame
from panoptic_polarnet.network.BEV_Unet import BEV_Unet
from panoptic_polarnet.network.ptBEV import ptBEVnet
from panoptic_polarnet.network.instance_post_processing import get_panoptic_segmentation


class ScatterMax(nn.Module):

    def forward(self, src: torch.Tensor, index: torch.Tensor):
        # src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
        # index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        out, argmax = scatter_max(src, index, dim=-1, out=src)
        return out, argmax


@symbolic_helper.parse_args("v", "v", "i", "v", "i", "i")
def symbolic_scatter_max(
    g, src, index, dim=-1, out=None, dim_size=None, fill_value=None
):
    return (
        index,
        g.op("ScatterElements", out, index, src, axis_i=dim, reduction_s="max"),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for converting model to onnx format"
    )
    parser.add_argument(
        "-c", "--configs", default="configs/SemanticKITTI_model/Panoptic-PolarNet.yaml"
    )
    parser.add_argument(
        "-f", "--frame", default="/data/sequences/10/velodyne/000000.bin"
    )
    parser.add_argument("-o", "--onnx-filename", default="panoptic_polarnet.onnx")
    args = parser.parse_args()

    with open(args.configs, "r") as s:
        new_args = yaml.safe_load(s)

    grid_size = new_args["dataset"]["grid_size"]
    grid_size = np.asarray(grid_size)
    panoptic_proc = PanopticLabelGenerator(
        grid_size, sigma=new_args["dataset"]["gt_generator"]["sigma"], polar=True
    )
    data_tuple, thing_list = process_one_frame(args.frame, grid_size, panoptic_proc)

    pretrained_model = "pretrained_weight/Panoptic_SemKITTI_PolarNet.pt"
    compression_model = new_args["dataset"]["grid_size"][2]
    grid_size = new_args["dataset"]["grid_size"]
    visibility = new_args["model"]["visibility"]
    pytorch_device = torch.device("cuda:0")
    if new_args["model"]["polar"]:
        fea_dim = 9
        circular_padding = True
    else:
        fea_dim = 7
        circular_padding = False

    # !!! prepare miou fun
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    # prepare model
    my_BEV_model = BEV_Unet(
        n_class=len(unique_label),
        n_height=compression_model,
        input_batch_norm=True,
        dropout=0.5,
        circular_padding=circular_padding,
        use_vis_fea=visibility,
    )
    my_model = ptBEVnet(
        my_BEV_model,
        pt_model="pointnet",
        grid_size=grid_size,
        fea_dim=fea_dim,
        max_pt_per_encode=256,
        out_pt_fea_dim=512,
        kernal_size=1,
        pt_selection="random",
        # pt_selection="farthest",
        fea_compre=compression_model,
        shuffle_data=False,
    )
    print(os.path.exists(pretrained_model))
    if os.path.exists(pretrained_model):
        my_model.load_state_dict(torch.load(pretrained_model))
    pytorch_total_params = sum(p.numel() for p in my_model.parameters())
    print("params: ", pytorch_total_params)
    my_model.to(pytorch_device)
    my_model.eval()

    with torch.no_grad():

        test_vox_fea, _, _, _, test_grid, _, _, test_pt_fea = data_tuple

        test_vox_fea = torch.from_numpy(
            np.expand_dims(test_vox_fea, axis=0).astype(np.float32)
        )
        test_grid = np.expand_dims(test_grid, axis=0)
        test_pt_fea = np.expand_dims(test_pt_fea, axis=0)

        # predict
        test_vox_fea_ten = test_vox_fea.to(pytorch_device)
        test_pt_fea_ten = [
            torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device)
            for i in test_pt_fea
        ]
        test_grid_ten = [
            torch.from_numpy(i[:, :2]).to(pytorch_device) for i in test_grid
        ]

        # model
        predict_labels, center, offset = my_model(
            test_pt_fea_ten, test_grid_ten, test_vox_fea_ten
        )

        # convert to onnx
        torch.onnx.register_custom_op_symbolic(
            "torch_scatter::scatter_max", symbolic_scatter_max, 18
        )
        # torch.onnx.export(
        #     my_model,
        #     (test_pt_fea_ten, test_grid_ten, test_vox_fea_ten),
        #     args.onnx_filename,
        #     export_params=True,
        #     opset_version=19,
        #     do_constant_folding=False,
        #     # output_names=["output"],
        #     # dynamic_axes={'cat_pt_fea' : {0 : 'n_points'},
        #     #               'unq_inv' : {0 : 'n_points'},
        #     #                     'output' : {0 : 'n_scattered'}},
        # )
        torch.onnx.export(
            my_model.PPmodel,
            (torch.randn(121184, 9).to(pytorch_device), ),
            "pp_model.onnx",
            export_params=True,
            opset_version=18,
            do_constant_folding=False,
            input_names=["cat_pt_fea"],
            output_names=["output"],
            dynamic_axes={
                'cat_pt_fea' : {0 : 'n_points'},
                'output' : {0 : 'n_points'},
            },
        )
        pp_onnx_model = onnx.load("pp_model.onnx")
        onnx.checker.check_model(pp_onnx_model, full_check=True)

        torch.onnx.export(
            my_model.BEV_model,
            (torch.randn(3, 64, 480, 360).to(pytorch_device), ),
            "bev_model.onnx",
            export_params=True,
            opset_version=18,
            do_constant_folding=False,
            input_names=["bev_data"],
            output_names=["sem_prediction", "center", "offset"],
            dynamic_axes={
                'bev_data' : {0 : 'batch_size'},
                'sem_prediction' : {0 : 'batch_size'},
                'center' : {0 : 'batch_size'},
                'offset' : {0 : 'batch_size'},
            },
        )
        bev_onnx_model = onnx.load("bev_model.onnx")
        onnx.checker.check_model(bev_onnx_model, full_check=True)


        import onnxruntime as ort
        bev_onnx_model_session = ort.InferenceSession("bev_model.onnx")
        pp_onnx_model_session = ort.InferenceSession("pp_model.onnx")
        predict_labels, center, offset = my_model(
            test_pt_fea_ten, test_grid_ten, test_vox_fea_ten,
            pp_model = lambda x: torch.from_numpy(
                pp_onnx_model_session.run(None, {"cat_pt_fea": x.detach().cpu().numpy()})[0]
            ).to(pytorch_device),
            bev_model = lambda x: bev_onnx_model_session.run(None, {"bev_data": x.detach().cpu().numpy()}),
        )
        print(predict_labels.shape)
        print(center.shape)
        print(offset.shape)


        # get labels
        count = 0

        # get foreground_mask
        for_mask = torch.zeros(
            1, grid_size[0], grid_size[1], grid_size[2], dtype=torch.bool
        ).to(pytorch_device)
        for_mask[
            0, test_grid[count][:, 0], test_grid[count][:, 1], test_grid[count][:, 2]
        ] = True


        # post processing
        panoptic_labels, center_points = get_panoptic_segmentation(
            torch.unsqueeze(torch.from_numpy(predict_labels[0]), 0),#torch.unsqueeze(predict_labels[0], 0),
            torch.unsqueeze(torch.from_numpy(center[0]), 0),#torch.unsqueeze(center[0], 0),
            torch.unsqueeze(torch.from_numpy(offset[0]), 0),#torch.unsqueeze(offset[0], 0),
            thing_list,
            threshold=0.1,  # args['model']['post_proc']['threshold'],
            nms_kernel=5,  # args['model']['post_proc']['nms_kernel'],
            top_k=100,  # args['model']['post_proc']['top_k'],
            polar=True,
            foreground_mask=for_mask.cpu(),
        )
        panoptic_labels = panoptic_labels.cpu().detach().numpy().astype(np.uint32)

        panoptic = panoptic_labels[
            0, test_grid[count][:, 0], test_grid[count][:, 1], test_grid[count][:, 2]
        ]
        panoptic.tofile("test_onnx.label")

