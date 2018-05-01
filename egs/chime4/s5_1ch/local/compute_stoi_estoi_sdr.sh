#!/bin/bash
# Copyright 2017 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

# This script creates the list of enhanced files and reference files and calls the 
# matlab script "stoi_estoi_sdr.m" to get STOI, eSTOI and SDR scores
# Eg. local/compute_stoi_estoi_sdr.sh --njobs 10 blstm_gev enhan/blstm_gev local/nn-gev/data/audio/16kHz/isolated_ext  

. ./cmd.sh
. ./path.sh
set -e
set -u
set -o pipefail

njobs=10
cmd=run.pl

. utils/parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 3)"
   echo "Usage: local/compute_stoi_estoi_sdr.sh [options] <enhancement-method> <enhancement-directory> <chime-rir-directory>"
   echo "options"
   echo "  --njobs <njobs>                          # number of parallel jobs"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   exit 1;
fi

enhancement_method=$1
enhancement_directory=$2
chime_rir_directory=$3

expdir=exp/compute_stoi_estoi_sdr_${enhancement_method}
mkdir -p $expdir
ls $chime_rir_directory/dt05_*/*CH5.Clean.wav > $expdir/original_list
ls $enhancement_directory/dt05_*simu/*.wav > $expdir/enhanced_list
$cmd $expdir/compute_stoi_estoi_sdr_dt05.log matlab -nodisplay -nosplash -r "addpath('local'); stoi_estoi_sdr($njobs,'$enhancement_method','$expdir','dt05');exit"
ls $chime_rir_directory/et05_*/*CH5.Clean.wav > $expdir/original_list
ls $enhancement_directory/et05_*simu/*.wav > $expdir/enhanced_list
$cmd $expdir/compute_stoi_estoi_sdr_et05.log matlab -nodisplay -nosplash -r "addpath('local'); stoi_estoi_sdr($njobs,'$enhancement_method','$expdir','et05');exit"
