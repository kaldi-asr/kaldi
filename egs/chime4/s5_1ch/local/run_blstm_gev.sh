#!/bin/bash
# Copyright 2017 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

. ./cmd.sh
. ./path.sh

# Config:
nj=10
cmd=run.pl
track=6
. utils/parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Wrong #arguments ($#, expected 4)"
   echo "Usage: local/run_blstm_gev.sh [options] <chime4-dir> <chime3-dir> <wav-out-dir> <enhancement-type>"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --track <track>            # Chime data to use (1, 2 or 6)"
   exit 1;
fi

sdir=$1
chime3_dir=$2
odir=$3
enhancement_type=$4

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

miniconda_dir=$HOME/miniconda3/
if [ ! -d $miniconda_dir ]; then
    echo "$miniconda_dir does not exist. Please run '../../../tools/extras/install_miniconda.sh' and '../../../tools/extras/install_chainer.sh';"
    echo "For chainer install, you have to use cudnn <= 5.1."
    echo "In the CLSP cluster, $HOME/miniconda3/bin/python -m pip uninstall chainer; . /home/asubraman/.bash_profile_cuda_path; $HOME/miniconda3/bin/python -m pip install chainer==1.16.0 -vvvv --no-cache" && exit 1;
fi

# check if chainer is installed
result=`$HOME/miniconda3/bin/python -c "\
try:
    import chainer
    print('1')
except ImportError:
    print('0')"`

if [ "$result" == "1" ]; then
    echo "Chainer is installed"
else
    echo "Chainer is not installed. Please run ../../../tools/extras/install_chainer.sh"
    echo "For chainer install, you have to use cudnn <= 5.1."
    echo "In the CLSP cluster, $HOME/miniconda3/bin/python -m pip uninstall chainer; . /home/asubraman/.bash_profile_cuda_path; $HOME/miniconda3/bin/python -m pip install chainer==1.16.0 -vvvv --no-cache" && exit 1;   
fi

if [ ! -d local/nn-gev ]; then
    cd local/
    git clone https://github.com/fgnt/nn-gev.git
    cd nn-gev/
    git apply ../fix_read_sim_from_different_directory.patch
    cd ../../
else
    cd local/nn-gev/
    git checkout .
    git apply ../fix_read_sim_from_different_directory.patch
    cd ../../
fi

mkdir -p $odir
set +e
nIsolatedDirs=`ls local/nn-gev/data/audio/16kHz/isolated/ 2>/dev/null | wc -l`
nIsolatedExtDirs=`ls local/nn-gev/data/audio/16kHz/isolated_ext/ 2>/dev/null | wc -l`
set -e
if [[ "$nIsolatedDirs" -ne 12 || "$nIsolatedExtDirs" -ne 12 ]];then
   echo "generating simulation data and storing in local/nn-gev/data"
   $cmd $odir/simulation.log matlab -nodisplay -nosplash -r "addpath('local'); CHiME3_simulate_data_patched_parallel(1,$nj,'$sdir','$chime3_dir');exit"
else
   echo "Didn't run Matlab simulation. Using existing data in local/nn-gev/data/audio/"
fi

echo "Training a BLSTM-based mask network and enhancing signals with mask-based GEV beamformer"
$cuda_cmd $odir/beamform.log local/run_nn-gev.sh $sdir $odir $enhancement_type $track
