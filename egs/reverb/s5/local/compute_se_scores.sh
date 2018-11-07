#!/bin/bash
# Copyright 2018 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

# This script computes the dereverberation scores given in REVERB challenge
# Eg. local/compute_se_scores.sh --nch 8 /export/corpora5/REVERB_2014/REVERB ${PWD}/wav ${PWD}/local 

. ./cmd.sh
. ./path.sh
set -e
set -u
set -o pipefail

cmd=run.pl
nch=8

. utils/parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 3)"
   echo "Usage: local/compute_se.sh [options] <reverb_data> <enhancement-directory> <pesq-directory>"
   echo "options"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --nch <nch>                              # nch of WPE to use for computing SE scores"
   exit 1;
fi

reverb_data=$1
enhancement_directory=$2
pesqdir=$3
enhancement_directory_sim=$enhancement_directory/WPE/${nch}ch/REVERB_WSJCAM0_dt/data/
enhancement_directory_real=$enhancement_directory/WPE/${nch}ch/MC_WSJ_AV_Dev/

expdir=${PWD}/exp/compute_se_${nch}ch
pushd local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/evaltools
$cmd $expdir/compute_se_real.log matlab -nodisplay -nosplash -r "addpath('SRMRToolbox'); score_RealData('$reverb_data','$enhancement_directory_real');exit"
$cmd $expdir/compute_se_sim.log matlab -nodisplay -nosplash -r "addpath('SRMRToolbox'); score_SimData('$reverb_data','$enhancement_directory_sim','$pesqdir');exit"
popd
mv local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/scores $expdir/
