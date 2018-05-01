#!/bin/bash
# Copyright 2017 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

# This script creates the average PESQ score of files in an enhanced directory with corresponding 
# files in a reference directory.
# Expects the PESQ third party executable in "local/PESQ"
# PESQ source was dowloaded and compiled using "local/download_se_eval_tool.sh" 
# Eg. local/compute_pesq.sh blstm_gev enhan/blstm_gev local/nn-gev/data/audio/16kHz/isolated_ext $PWD

set -e
set -u
set -o pipefail

if [ $# != 4 ]; then
   echo "Wrong #arguments ($#, expected 4)"
   echo "Usage: local/compute_pesq.sh <enhancement-method> <enhancement-directory> <chime-rir-directory> <modeldir>"
   exit 1;
fi

enhancement_method=$1
enhancement_directory=$2
chime_rir_directory=$3
modeldir=$4

expdir=$modeldir/exp/compute_pesq_${enhancement_method}
mkdir -p $expdir
pushd $expdir
ls $enhancement_directory/et05_*_simu/*.wav > $expdir/et05_files
ls $enhancement_directory/dt05_*_simu/*.wav > $expdir/dt05_files

for set in "dt05" "et05"
do
declare -i n_files=0
t_mos=0
avg_mos=0
  while read filename; do
    n_files=$n_files+1
    target_filename=`echo $filename | rev | cut -d"/" -f1 | rev`
    speaker=`echo $target_filename | cut -d"_" -f1`
    utt_id=`echo $target_filename | cut -d"_" -f2`
    noise_cap=`echo $target_filename | cut -d"_" -f3 | cut -d"." -f1`
    noise=`echo "$noise_cap" | awk '{ print tolower($1) }'`
    temp=`$modeldir/local/PESQ +16000 ../../$chime_rir_directory/"$set"_"$noise"_simu/"$speaker"_"$utt_id"_"$noise_cap".CH5.Clean.wav $filename`
    pesq_score=`echo $temp | rev | cut -d " " -f1 | rev`
    t_mos=$(awk "BEGIN {print $t_mos+$pesq_score; exit}")
  done <$expdir/"$set"_files
avg_mos=$(awk "BEGIN {print $t_mos/$n_files; exit}")
echo $avg_mos>"$expdir"/pesq_"$set"
done
popd
