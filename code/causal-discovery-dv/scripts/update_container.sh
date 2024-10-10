#! /bin/bash

echo "<<UPDATING DOCKER IMAGE...>>"
# docker build -f $CDDV_SOURCE_DIR/docker/base/Dockerfile -t causal-discovery-dv/base --rm --progress=plain $CDDV_SOURCE_DIR && docker image prune -f
docker build -f $CDDV_SOURCE_DIR/docker/base/Dockerfile -t causal-discovery-dv/base --rm $CDDV_SOURCE_DIR && docker image prune -f
docker build -f $CDDV_SOURCE_DIR/docker/manager/Dockerfile -t causal-discovery-dv/manager --rm $CDDV_SOURCE_DIR && docker image prune -f