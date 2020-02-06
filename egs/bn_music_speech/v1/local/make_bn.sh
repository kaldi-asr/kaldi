#!/usr/bin/env bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This script, called by ../run.sh, creates the HUB4 Broadcast News
# data directory. The required datasets can be found at:
#   https://catalog.ldc.upenn.edu/LDC97S44
#   https://catalog.ldc.upenn.edu/LDC97T22

set -e
sph_dir=$1
transcript_dir=$2
data_dir=$3
tmp_dir=local/bn.tmp

# These parameters are used when refining the annotations.
# A higher frames_per_second provides better resolution at the
# frame boundaries. Set min_seg to control the minimum length of the
# final segments. It seems that the original annotations for segments
# below half a second are not very accurate, so we test only on segments
# longer than this.
frames_per_sec=100
min_seg=0.5

rm -rf local/bn.tmp
mkdir local/bn.tmp

echo "$0: preparing annotations..."
local/make_annotations_bn.py ${transcript_dir} ${tmp_dir}
echo "$0: Removing overlapping annotations..."
local/refine_annotations_bn.py ${tmp_dir} ${frames_per_sec} ${min_seg}
echo "$0: Preparing broadcast news data directories ${data_dir}/bn..."
local/make_bn.py ${sph_dir} ${tmp_dir}

mkdir -p ${data_dir}/bn
cp ${tmp_dir}/wav.scp ${data_dir}/bn/
cp ${tmp_dir}/utt2spk ${data_dir}/bn/
cp ${tmp_dir}/segments ${data_dir}/bn/
rm -rf local/bn.tmp
utils/fix_data_dir.sh data/bn
