#!/bin/bash
# Copyright 2017 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

. ./cmd.sh
. ./path.sh

# Config:
nj=10
cmd=run.pl
. utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Wrong #arguments ($#, expected 2)"
   echo "Usage: local/run_beamform_blstm_gev_6ch_track.sh [options] <chime-dir> <wav-out-dir>"
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

miniconda_dir=$HOME/miniconda3/
if [ ! -d $miniconda_dir ]; then
  echo "$miniconda_dir does not exist. Please run '../../../tools/extras/install_miniconda.sh' and '../../../tools/extras/install_chainer.sh';" && exit 1;
fi

if [ !-d local/nn-gev ]; then
    cd local/
    git clone https://github.com/sas91/nn-gev.git
    cd ../
fi

mkdir -p $odir
$cmd $odir/simulation.log matlab -nodisplay -nosplash -r "addpath('local'); CHiME3_simulate_data_patched_parallel(1,$nj,'$sdir');exit"
$cuda_cmd $odir/beamform.log local/run_nn-gev.sh $sdir $odir
