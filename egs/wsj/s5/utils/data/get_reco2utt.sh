#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

if [ $# -ne 1 ]; then
  echo "This script creates a reco2utt file in the data directory, "
  echo "which is analogous to spk2utt file but with the first column "
  echo "as recording instead of speaker."
  echo "Usage: get_reco2utt.sh <data>"
  echo " e.g.: get_reco2utt.sh data/train"
  exit 1
fi

data=$1

if [ ! -s $data/segments ]; then
  utils/data/get_segments_for_data.sh $data > $data/segments
fi

cut -d ' ' -f 1,2 $data/segments | utils/utt2spk_to_spk2utt.pl > $data/reco2utt
