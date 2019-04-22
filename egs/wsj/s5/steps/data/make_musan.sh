#!/bin/bash
# Copyright 2015   David Snyder
#           2019   Phani Sankar Nidadavolu
# Apache 2.0.
#
# This script creates the MUSAN data directory.
# Consists of babble, music and noise files.
# Used to create augmented data
# The required dataset is freely available at http://www.openslr.org/17/

set -e
use_vocals=true
sampling_rate=16000
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
    echo USAGE: $0 input_dir output_dir
    echo input_dir is the path where the original musal corpus is located
    echo e.g: $0 /export/corpora/JHU/musan data
    exit 1;
fi

in_dir=$1
data_dir=$2

if [ -d $data_dir/musan_music ] && [ -d $data_dir/musan_noise ] && [ -d $data_dir/musan_speech ]; then
    echo "Required musan directories already exists, not making them again"
else
    mkdir -p local/musan.tmp

    # The below script will create the musan corpus
    steps/data/make_musan.py --use-vocals ${use_vocals} \
            --sampling-rate ${sampling_rate} \
            ${in_dir} ${data_dir}/musan || exit 1;

    utils/fix_data_dir.sh ${data_dir}/musan

    grep "music" ${data_dir}/musan/utt2spk > local/musan.tmp/utt2spk_music
    grep "speech" ${data_dir}/musan/utt2spk > local/musan.tmp/utt2spk_speech
    grep "noise" ${data_dir}/musan/utt2spk > local/musan.tmp/utt2spk_noise

    utils/subset_data_dir.sh --utt-list local/musan.tmp/utt2spk_music \
        ${data_dir}/musan ${data_dir}/musan_music
    utils/subset_data_dir.sh --utt-list local/musan.tmp/utt2spk_speech \
        ${data_dir}/musan ${data_dir}/musan_speech
    utils/subset_data_dir.sh --utt-list local/musan.tmp/utt2spk_noise \
        ${data_dir}/musan ${data_dir}/musan_noise

    utils/fix_data_dir.sh ${data_dir}/musan_music
    utils/fix_data_dir.sh ${data_dir}/musan_speech
    utils/fix_data_dir.sh ${data_dir}/musan_noise

    rm -rf local/musan.tmp
fi

for name in speech noise music; do
    utils/data/get_reco2dur.sh ${data_dir}/musan_${name}
done
