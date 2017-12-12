#!/bin/bash

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
python local/nn-gev-master/train.py --chime_dir=$sdir --gpu $gpu_id $odir BLSTM
beamform.sh --gpu $gpu_id $sdir $odir $odir/BLSTM_model/best.nnet BLSTM
