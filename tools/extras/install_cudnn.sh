#!/bin/bash

# https://developer.nvidia.com/rdp/cudnn-download

if [ ! -f cudnn-7.0-linux-x64-v4.0-prod.tgz ]; then
  echo This script cannot install cuDNN in a completely automatic
  echo way because you need to register on the website before downloading.
  echo Please download cuDNN from https://developer.nvidia.com/rdp/cudnn-download
  echo put it in ./cudnn-7.0-linux-x64-v4.0-prod.tgz, then run this script.
  exit 1
fi

tar -xvzf cudnn-7.0-linux-x64-v4.0-prod.tgz --transform 's/cuda/cudnn/'

echo >&2 "Installation of cuDNN finished successfully"

