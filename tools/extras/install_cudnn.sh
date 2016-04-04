#!/bin/bash

# https://developer.nvidia.com/rdp/cudnn-download

if [ ! -f cudnn-7.0-linux-x64-v4.0-prod.tgz ]; then
  wget -T 10 -t 3 http://developer.download.nvidia.com/compute/redist/cudnn/v4/cudnn-7.0-linux-x64-v4.0-prod.tgz || exit 1;
fi

tar -xvzf cudnn-7.0-linux-x64-v4.0-prod.tgz --transform 's/cuda/cudnn/'

echo >&2 "Installation of cuDNN finished successfully"

