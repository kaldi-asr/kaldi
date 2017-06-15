#!/bin/bash

# Copyright 2017 Johns Hopkins University (author: Chun-Chieh "Jonathan" Chang)

# This script loads the training and test data for
# Imagenet 2012 Task 1 or 3 Classification

# Currently the script is set to run the one for Task 3
# To change modify the paths to the correct directories

[ -f ./path.sh ] && . ./path.sh;

# Path to imagenet directory
# Requires the datasets for Task 1:
# "development kit", "training images", "validation images", and "test images"
dl_dir=/export/b18/imagenet_2012/

# Various other paths
devkit_dir=$dl_dir/devkit_t3
train_dir=$dl_dir/train_t3
val_dir=$dl_dir/val
test_dir=$dl_dir/test

# Various tar files
devkit_tar=ILSVRC2012_devkit_t3.tar.gz
train_tar=ILSVRC2012_img_train_t3.tar
val_tar=ILSVRC2012_img_val.tar
test_tar=ILSVRC2012_img_test.tar

# Extra
# For when running the task 3
# the devkit used for classes.txt still needs to be from task 1
devkit_dir_t12=$dl_dir/devkit_t12
devkit_tar_t12=ILSVRC2012_devkit_t12.tar.gz

# Check if dataset is downloaded 
if [ ! -d $dl_dir ] || \
     [ ! -f $dl_dir/$devkit_tar ] || \
     [ ! -f $dl_dir/$train_tar ] || \
     [ ! -f $dl_dir/$val_tar ] || \
     [ ! -f $dl_dir/$test_tar ] || \
     [ ! -f $dl_dir/$devkit_tar_t12 ]; then
  echo Need to download ImageNet2012 dataset first. Need tar for devkit train val and test data.
  exit 1
else
  if [ ! -d $devkit_dir ]; then
    mkdir -p $devkit_dir
    tar -xvzf $dl_dir/$devkit_tar -C $devkit_dir || exit 1
    # echo Missing devkit
    # exit 1
  fi

  if [ ! -d $devkit_dir_t12 ]; then
    mkdir -p $devkit_dir_t12
    tar -xvzf $dl_dir/$devkit_tar_t12 -C $devkit_dir || exit 1
    # echo Missing devkit_t12
    # exit 1
  fi

  if [ ! -d $train_dir ]; then
    mkdir -p $train_dir
    tar -xvf $dl_dir/$train_tar -C $train_dir || exit 1
    find $train_dir -name "*.tar" | \
    while read name; do mkdir -p "${name%.tar}"; tar -xvf "${name}" -C "${name%.tar}";done
    # echo Missing train
    # exit 1
  fi

  if [ ! -d $val_dir ]; then
    mkdir -p $val_dir
    tar -xvf $dl_dir/$val_tar -C $val_dir || exit 1
    # echo Missing val
    # exit 1
  fi

  if [ ! -d $test_dir ]; then
    mkdir -p $test_dir
    tar -xvf $dl_dir/$test_tar -C $test_dir || exit 1
    # echo Missing test
    # exit 1
  fi
fi

mkdir -p data/{train,val,test}/data

# Retrieve all the possible classes from the devkit .mat file
local/process_classes.py $devkit_dir_t12 $devkit_tar_t12 data --task 1 || exit 1

cp data/classes.txt data/train/classes.txt
cp data/classes.txt data/val/classes.txt
cp data/classes.txt data/test/classes.txt

echo 3 > data/train/num_channels
echo 3 > data/test/num_channels

# Process training data
#local/process_data.py $train_dir $devkit_dir $devkit_tar data/train --dataset train
local/process_data.py $train_dir $devkit_dir $devkit_tar data/train --dataset train | \
  copy-feats --compress=true --compression-method=7 \
  ark:- ark,scp:data/train/data/images.ark,data/train/images.scp || exit 1

# Process testing data
# Using validation data instead because testing data does not include ground truth
local/process_data.py $val_dir $devkit_dir $devkit_tar data/test \
  --dataset test --scale-size 256 --crop-size 224| \
  copy-feats --compress=true --compression-method=7 \
  ark:- ark,scp:data/test/data/images.ark,data/test/images.scp || exit 1



