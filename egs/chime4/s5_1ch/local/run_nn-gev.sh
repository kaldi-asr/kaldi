#!/bin/bash
# Copyright 2017 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

. ./cmd.sh
. ./path.sh

if [ $# != 2 ]; then
   echo "Wrong #arguments ($#, expected 2)"
   echo "Usage: local/run_nn-gev.sh <wav-in-dir> <wav-out-dir> <>"
   exit 1;
fi

gpu_id=1
case $(hostname -f) in
  *.clsp.jhu.edu) gpu_id=`free-gpu` ;; # JHU,
esac 
$HOME/miniconda3/bin/python local/nn-gev/train.py --chime_dir=$sdir/data --gpu $gpu_id local/nn-gev/data BLSTM
local/nn-gev/beamform.sh $sdir local/nn-gev/data $odir local/nn-gev/data/BLSTM_model/best.nnet BLSTM --gpu $gpu_id
