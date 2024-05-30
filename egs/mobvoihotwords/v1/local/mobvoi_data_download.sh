#!/bin/bash

# Copyright  2018-2020  Yiming Wang
#            2018-2020  Daniel Povey
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh

dl_dir=data/download

mkdir -p $dl_dir

dataset=mobvoi_hotword_dataset.tgz
resources=mobvoi_hotword_dataset_resources.tgz

# base url for downloads.
data_url=http://www.openslr.org/resources/87

if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  src_path=/export/fs04/a07/ywang/mobvoihotwords
else
  src_path=$dl_dir
fi

if [ ! -f $src_path/$dataset ] || [ ! -f $src_path/$resources ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi

  if [ ! -f $src_path/$dataset ]; then
    echo "$0: downloading data from $data_url/$dataset.  This may take some time, please be patient."
    if ! wget --no-check-certificate -O $dl_dir/$dataset $data_url/$dataset; then
      echo "$0: error executing wget $data_url/$dataset"
      exit 1;
    fi
  fi

  if [ ! -f $src_path/$resources ]; then
    if ! wget --no-check-certificate -O $dl_dir/$resources $data_url/$resources; then
      echo "$0: error executing wget $data_url/$resources"
      exit 1;
    fi
  fi
fi

if [ -d $dl_dir/$(basename "$dataset" .tgz) ]; then
  echo "Not extracting $(basename "$dataset" .tgz) as it is already there."
else
  echo "Extracting $dataset..."
  tar -xvzf $src_path/$dataset -C $dl_dir || exit 1;
  echo "Done extracting $dataset."
fi

if [ -d $dl_dir/$(basename "$resources" .tgz) ]; then
  echo "Not extracting $(basename "$dataset" .tar.gz) as it is already there."
else
  echo "Extracting $resources..."
  tar -xvzf $src_path/$resources -C $dl_dir || exit 1;
  echo "Done extracting $resources."
fi

exit 0
