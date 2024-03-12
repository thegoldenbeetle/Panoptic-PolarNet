# PanopticPolarNet

## Environment:

Download dataset: https://www.nuscenes.org/

Run docker:
- docker-compose -f ./docker-compose.yaml build
- docker-compose -f ./docker-compose.yaml run --rm workspace 

Run jupyter inside docker:
- jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

Forward X11 inside docker:
- xhost +local:docker

## Usage:

Train the model
`python3 -m panoptic_polarnet.train` 

Evaluate
`python3 -m panoptic_polarnet.test_pretrain` 

Inference
`python3 -m panoptic_polarnet.inference` 

Visualize
`python3 -m panoptic_polarnet.visualize_pointclouds` 


## TODO:

- [x] Visualize pointcloud: frame
- [x] Visualize pointcloud: sequence
- [x] Visualize labels and predictions with different colors
- [x] Run inference on one frame on torch (visualize predictions + get fps)
- [ ] Convert model to onnx and run inference on it +/-
- [ ] Run training +/-
- [ ] Analyze current performance +/- with different conditions

- [ ] Run inference on sequence with ros?

