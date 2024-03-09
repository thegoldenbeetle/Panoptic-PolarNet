Download dataset:
https://www.nuscenes.org/

Run docker:

- docker-compose -f ./docker-compose.yaml build
- docker-compose -f ./docker-compose.yaml run --rm workspace

Run jupyter inside docker:

- jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

X11 inside docker:

- xhost +local:docker


### TODO:

1. + Visualize pointcloud frame
2. + Visualize pointcloud kitti sequence
3. Visualize labels
4. Run inference on one frame (visualize predictions + get fps)
5. Convert model to onnx
6. Run inference on sequence ?
7. Run training


