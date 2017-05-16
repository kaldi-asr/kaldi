#! /bin/bash

# Copyright  2016  Vimal Manohar
# Apache 2.0

# This script prepares MUSAN music corpus for data perturbation.

if [ $# -ne 2 ]; then
  echo "Usage: $0 <MUSAN-SRC-DIR> <dir>"
  echo " e.g.: $0 /export/corpora/JHU/musan RIRS_NOISES/music"
  exit 1
fi

SRC_DIR=$1
dir=$2

mkdir -p $dir

local/segmentation/make_musan_music.py $SRC_DIR $dir/wav.scp

wav-to-duration scp:$dir/wav.scp ark,t:$dir/reco2dur
steps/data/split_wavs_randomly.py $dir/wav.scp $dir/reco2dur \
  $dir/split_utt2dur $dir/split_wav.scp

awk '{print $1" "int($2*100)}' $dir/split_utt2dur > $dir/split_utt2num_frames
steps/data/wav_scp2noise_list.py $dir/split_wav.scp $dir/music_list
