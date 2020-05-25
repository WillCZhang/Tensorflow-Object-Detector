#!/bin/bash

# Build Object Detector
docker build -t object-detector ./detector # TODO
docker build -t object-detector-detect -f ./detector/Dockerfile-Detect ./detector
