Download dataset:
<!-- https://www.nuscenes.org/ -->
Run docker:

docker-compose -f ./docker-compose.yaml build
docker-compose -f ./docker-compose.yaml run --rm workspace

Run jupyter inside docker:

jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html



