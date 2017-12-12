#!/bin/bash

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

if [ !-d local/nn-gev-master ]; then
    cd local/
    git clone https://github.com/fgnt/nn-gev.git
    cd ../
fi

mkdir -p $odir
$cmd $odir/simulation.log matlab -nodisplay -nosplash -r "addpath('local'); CHiME3_simulate_data_patched_parallel(1,$njobs,'$chime4_dir');exit"

case $(hostname -f) in
  *.clsp.jhu.edu) gpu_id=`free-gpu` ;; # JHU,
esac
$cuda_cmd $odir/beamform.log local/run_nn-gev.sh $sdir $odir
