#!/bin/bash

xhost +local:docker

docker run -it --rm \
  --env DISPLAY=$DISPLAY \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/video0 \
  --name reflection \
  reflection