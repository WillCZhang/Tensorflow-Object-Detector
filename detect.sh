#!/bin/bash
set -e

. configuration

docker run -it --env-file detector/env -p 8888:8888 -v $DATA_PATH:/data -v $MODEL_PATH:/model --rm object-detector-detect
