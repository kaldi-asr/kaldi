#!/bin/bash
# Copyright 2017 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

. ./cmd.sh
. ./path.sh

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 2)"
   echo "Usage: local/run_nn-gev.sh <wav-in-dir> <wav-out-dir> <enhancement-type>"
   exit 1;
fi

sdir=$1
odir=$2
enhancement_type=$3

gpu_id=1
case $(hostname -f) in
  *.clsp.jhu.edu) gpu_id=`free-gpu` ;; # JHU,
esac 

echo "training a BLSTM mask network"
$HOME/miniconda3/bin/python local/nn-gev/train.py --chime_dir=$sdir/data --gpu $gpu_id local/nn-gev/data BLSTM
echo "enhancing signals with mask-based GEV beamformer"
local/nn-gev/beamform.sh $sdir/data local/nn-gev/data $odir local/nn-gev/data/BLSTM_model/best.nnet BLSTM --gpu $gpu_id --single $enhancement_type
