#!/usr/bin/env bash
set -e

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# This is modified from the script in standard Kaldi recipe to account
# for the way the WSJ data is structured on the Edinburgh systems.
# - Arnab Ghoshal, 29/05/12

# Modified from the script for CHiME2 baseline
# Shinji Watanabe 02/13/2015
# Modified to use data of six channels
# Szu-Jui Chen 09/29/2017

# Config:
eval_flag=true # make it true when the evaluation data are released

. utils/parse_options.sh || exit 1;

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <corpus-directory>\n\n" `basename $0`
  echo "The argument should be a the top-level Chime4 directory."
  echo "It is assumed that there will be a 'data' subdirectory"
  echo "within the top-level corpus directory."
  exit 1;
fi

echo "$0 $@"  # Print the command line for logging

audio_dir=$1/data/audio/16kHz/isolated
trans_dir=$1/data/transcriptions

echo "extract all channels (CH[1-6].wav) for noisy data"

dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils

. ./path.sh # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
  echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
  exit 1;
fi

if $eval_flag; then
list_set="tr05_simu_noisy dt05_simu_noisy et05_simu_noisy"
else
list_set="tr05_simu_noisy dt05_simu_noisy"
fi

cd $dir

find $audio_dir -name '*CH[1-6].wav' | grep 'tr05_bus_simu\|tr05_caf_simu\|tr05_ped_simu\|tr05_str_simu' | sort -u > tr05_simu_noisy.flist
find $audio_dir -name '*CH[1-6].wav' | grep 'dt05_bus_simu\|dt05_caf_simu\|dt05_ped_simu\|dt05_str_simu' | sort -u > dt05_simu_noisy.flist
if $eval_flag; then
find $audio_dir -name '*CH[1-6].wav' | grep 'et05_bus_simu\|et05_caf_simu\|et05_ped_simu\|et05_str_simu' | sort -u > et05_simu_noisy.flist
fi

# make a dot format from json annotation files
cp $trans_dir/dt05_simu.dot_all dt05_simu.dot
if $eval_flag; then
cp $trans_dir/et05_simu.dot_all et05_simu.dot
fi

# make a scp file from file list
for x in $list_set; do
    cat $x.flist | awk -F'[/]' '{print $NF}'| sed -e 's/\.wav/_SIMU/' > ${x}_wav.id.temp
    cat ${x}_wav.id.temp | awk -F'_' '{print $3}' | awk -F'.' '{print $2}' > $x.ch
    cat ${x}_wav.id.temp | awk -F'_' '{print $1}' > $x.part1
    cat ${x}_wav.id.temp | sed -e 's/^..._//' > $x.part2
    paste -d"_" $x.part1 $x.ch $x.part2 > ${x}_wav.ids
    paste -d" " ${x}_wav.ids $x.flist | sort -t_ -k1,1 -k3 > ${x}_wav.scp.temp
done

# make a transcription from dot
# simulation training data extract dot file from original WSJ0 data
# since it is generated from these data
if [ ! -e dot_files.flist ]; then
  echo "Could not find $dir/dot_files.flist files, first run local/clean_wsj0_data_prep.sh";
  exit 1;
fi
cat tr05_simu_noisy_wav.scp.temp | awk -F'[_]' '{print $3}' | tr '[A-Z]' '[a-z]' \
    | $local/find_noisy_transcripts.pl dot_files.flist | cut -f 2- -d" " > tr05_simu_noisy.txt
cat tr05_simu_noisy_wav.scp.temp | cut -f 1 -d" " > tr05_simu_noisy.ids
paste -d" " tr05_simu_noisy.ids tr05_simu_noisy.txt | sort -t_ -k1,1 -k3 > tr05_simu_noisy.trans1
# dt05 and et05 simulation data are generated from the CHiME4 booth recording
# and we use CHiME4 dot files
cat dt05_simu.dot | sed -e 's/(\(.*\))/\1/' | awk '{print $NF ".CH1_SIMU"}'> dt05_simu_noisy.ids
cat dt05_simu.dot | sed -e 's/(.*)//' > dt05_simu_noisy.txt
paste -d" " dt05_simu_noisy.ids dt05_simu_noisy.txt | \
awk '{print}{sub(/CH1/, "CH2",$0);print}{sub(/CH2/, "CH3",$0);print}{sub(/CH3/, "CH4",$0);print}{sub(/CH4/, "CH5",$0);print}{sub(/CH5/, "CH6",$0);print}' | \
sort -k 1 > dt05_simu_noisy.trans1
if $eval_flag; then
cat et05_simu.dot | sed -e 's/(\(.*\))/\1/' | awk '{print $NF ".CH1_SIMU"}'> et05_simu_noisy.ids
cat et05_simu.dot | sed -e 's/(.*)//' > et05_simu_noisy.txt
paste -d" " et05_simu_noisy.ids et05_simu_noisy.txt | \
awk '{print}{sub(/CH1/, "CH2",$0);print}{sub(/CH2/, "CH3",$0);print}{sub(/CH3/, "CH4",$0);print}{sub(/CH4/, "CH5",$0);print}{sub(/CH5/, "CH6",$0);print}' | \
sort -k 1 > et05_simu_noisy.trans1
fi

# Do some basic normalization steps.  At this point we don't remove OOVs--
# that will be done inside the training scripts, as we'd like to make the
# data-preparation stage independent of the specific lexicon used.
noiseword="<NOISE>";
for x in $list_set;do
  cat ${x}_wav.scp.temp | awk '{print $1}' > $x.txt.part1
  cat $x.trans1 | awk '{$1=""; print $0}' | sed 's/^[ \t]*//g' > $x.txt.part2
  paste -d" " $x.txt.part1 $x.txt.part2 > $x.trans1
  cat $x.trans1 | $local/normalize_transcript.pl $noiseword \
    | sort > $x.txt || exit 1;
done

# Make the utt2spk and spk2utt files.
for x in $list_set; do
  sort ${x}_wav.scp.temp > ${x}_wav.scp
  cat ${x}_wav.scp | awk -F'_' '{print $1"_"$2}' > $x.spk
  cat ${x}_wav.scp | awk '{print $1}' > $x.utt
  paste -d" " $x.utt $x.spk > $x.utt2spk
  cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;
done

# copying data to data/...
for x in $list_set; do
  mkdir -p ../../$x
  cp ${x}_wav.scp ../../$x/wav.scp || exit 1;
  cp ${x}.txt     ../../$x/text    || exit 1;
  cp ${x}.spk2utt ../../$x/spk2utt || exit 1;
  cp ${x}.utt2spk ../../$x/utt2spk || exit 1;
done

# clean up temp files
rm *.temp
rm *.part{1,2}

echo "Data preparation succeeded"
