#!/bin/bash

# Copyright 2019  Zili Huang
# Apache 2.0

# This script is part of VB resegmentation, it converts diagonal UBM and 
# ivector extractor to numpy array format  

# begin configuration section.
stage=0
cmd=run.pl

# end configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <gmm-model> <ivector-extractor-model> <VB-dir>"
  echo " Options:"
  echo "    --stage (0|1)  # start script from part-way through"
  echo "    --cmd (run.pl|queue.pl...)  # specify how to run the sub-processes"
  echo "e.g.:"
  echo "$0 exp/diag_ubm_1024/final.dubm exp/extractor_diag_c1024_i128/final.ie exp/VB"
  exit 1;
fi

gmm_model=$1
ivec_extractor=$2
VB_dir=$3

if [ $stage -le 0 ]; then
  # Dump the diagonal UBM model into txt format.
  "$train_cmd" $VB_dir/log/convert_diag_ubm.log \
    gmm-global-copy --binary=false \
      $gmm_model \
      $VB_dir/dubm.tmp || exit 1;

  # Dump the ivector extractor model into txt format.
  "$train_cmd" $VB_dir/log/convert_ie.log \
    ivector-extractor-copy --binary=false \
      $ivec_extractor \
      $VB_dir/ie.tmp || exit 1;
fi

if [ $stage -le 1 ]; then
  # Convert txt to numpy format
  python diarization/convert_VB_model.py $VB_dir/dubm.tmp $VB_dir/ie.tmp $VB_dir || exit 1;

  rm $VB_dir/dubm.tmp $VB_dir/ie.tmp || exit 1;
fi
