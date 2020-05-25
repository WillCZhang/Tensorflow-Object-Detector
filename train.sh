#!/bin/bash
set -e

. configuration

docker run -it --env-file detector/env -p 6006:6006 -v $DATA_PATH:/data -v $MODEL_PATH:/model --rm object-detector
