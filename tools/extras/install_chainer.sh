#!/usr/bin/env bash

# Installs chainer with nn-gev dependencies
# miniconda should be installed in $HOME/miniconda3/
# Download cuDNN from "https://developer.nvidia.com/rdp/cudnn-download" and extract in "$HOME/cuda/"
# and add their paths "export CFLAGS=-I$HOME/cuda/include" "export LDFLAGS=-L$HOME/cuda/lib64"

cudnn_dir=$HOME/cuda
cudnn_include_file=$cudnn_dir/include/cudnn.h
cudnn_lib_dir=$cudnn_dir/lib64
miniconda_dir=$HOME/miniconda3/

if [ ! -d $miniconda_dir ]; then
    echo "$miniconda_dir does not exist. Please run 'tools/extras/install_miniconda.sh" && exit 1;
fi

if [ ! -d $cudnn_lib_dir ] || [ ! -f $cudnn_include_file ]; then
    echo "cuDNN is not available. $cudnn_include_file and/or $cudnn_lib_dir are missing.
          Download cuDNN v5.1 for appropriate CUDA version (7.5 or 8.0) from 'https://developer.nvidia.com/rdp/cudnn-download'.
	  Check CUDA version using the command 'nvcc --version'
	  Place the include and lib directories in $cudnn_dir after download" && exit 1;
fi

cudnn_major=`cat $HOME/cuda/include/cudnn.h | grep CUDNN_MAJOR | head -1 | rev | cut -d " " -f1`
cudnn_minor=`cat $HOME/cuda/include/cudnn.h | grep CUDNN_MINOR | head -1 | rev | cut -d " " -f1`

if [ $cudnn_major -ne 5 ] || [ $cudnn_minor -ne 1 ]; then
    echo "cuDNN version in $cudnn_dir is not '5.1'. Please download v5.1"  && exit 1;
fi

$HOME/miniconda3/bin/python -m pip install --user chainer==1.16.0
