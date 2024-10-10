#! /bin/bash

echo "<<UPDATING DOCKER IMAGE...>>"
#docker build -f $CD_SOURCE_DIR/docker/base/Dockerfile -t causal-discovery/base --rm  --progress=plain $CD_SOURCE_DIR && docker image prune -f
docker build -f $CD_SOURCE_DIR/docker/base/Dockerfile -t causal-discovery/base --rm  $CD_SOURCE_DIR && docker image prune -f
docker build -f $CD_SOURCE_DIR/docker/manager/Dockerfile -t causal-discovery/manager --rm $CD_SOURCE_DIR && docker image prune -f
