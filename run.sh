#!/bin/sh
# groupmod --gid ${DOCKER_GID:-1000} lidar_user
# echo $DOCKER_UID  $DOCKER_GID
# usermod --uid ${DOCKER_UID:-1000} lidar_user

sudo -u lidar_user /bin/bash -c "${@:-bash}"
