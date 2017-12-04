#!/bin/bash

. ./cmd.sh
. ./path.sh

# Config:
nj=10
cmd=run.pl
gpu_id=1
. utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Wrong #arguments ($#, expected 2)"
   echo "Usage: local/run_beamform_blstm_gev_6ch_track.sh [options] <wav-in-dir> <wav-out-dir>"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   exit 1;
fi

sdir=$1
odir=$2

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

mkdir -p $odir
cat nn-gev-master/CHiME3/tools/simulation/CHiME3_simulate_data_patched_parallel.m | matlab

case $(hostname -f) in
  *.clsp.jhu.edu) gpu_id=`free-gpu` ;; # JHU,
esac 
python train.py --chime_dir=$sdir --gpu $gpu_id $odir BLSTM
beamform.sh $sdir $odir $odir/BLSTM_model/best.nnet BLSTM
