#!/usr/bin/env bash

export HOME=$PWD/tensorflow_build/

has_gpu=true

tf_source=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0-cp27-none-linux_x86_64.whl

if [ $has_gpu != "true" ]; then
  tf_source=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0-cp27-none-linux_x86_64.whl
fi

pip install --user $tf_source
