#!/bin/bash
# Copyright 2017 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

set -e
set -u
set -o pipefail

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 3)"
   echo "Usage: local/compute_PESQ.sh <enhancement-method> <enhancement-directory> <chime-RIR-directory>"
   exit 1;
fi

enhancement_method=$1
enhancement_directory=$2
chime_RIR_directory=$3

expdir=exp/compute_PESQ_${enhancement_method}
mkdir -p $expdir
ls $enhancement_directory/et05_*_simu/*.wav > $expdir/et05_files
ls $enhancement_directory/dt05_*_simu/*.wav > $expdir/dt05_files

for set in "dt05" "et05"
do
declare -i nFiles=0
tMOS=0
avg_mos=0
  while read filename; do
    nFiles=$nFiles+1
    target_filename=`echo $filename | rev | cut -d"/" -f1 | rev`
    speaker=`echo $target_filename | cut -d"_" -f1`
    utt_id=`echo $target_filename | cut -d"_" -f2`
    noise_cap=`echo $target_filename | cut -d"_" -f3 | cut -d"." -f1`
    noise=`echo "$noise_cap" | awk '{ print tolower($1) }'`
    temp=`local/PESQ +16000 $chime_RIR_directory/"$set"_"$noise"_simu/"$speaker"_"$utt_id"_"$noise_cap".CH5.Clean.wav $filename`
    pesq_score=`echo $temp | rev | cut -d " " -f1 | rev`
    tMOS=$(awk "BEGIN {print $tMOS+$pesq_score; exit}")
  done <$expdir/"$set"_files
avg_mos=$(awk "BEGIN {print $tMOS/$nFiles; exit}")
echo $avg_mos>"$expdir"/PESQ_"$set"
done
