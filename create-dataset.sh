#!/bin/bash
set -e

# This script will be executed inside the container

. configuration

# It will create three folders under $DATA_PATH
# images for labeled images
# training for TF_Record and training images
# testing for TF_Record and testing images
mkdir $DATA_PATH/images
cp -a $LABELED_IMAGE_PATH/. $DATA_PATH/images
docker run --env-file detector/env -v $DATA_PATH:/data --rm object-detector /app/create_dataset.sh
