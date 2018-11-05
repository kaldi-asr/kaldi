#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# Copy of egs/sre16/v1/local/make_musan.sh (commit e3fb7c4a0da4167f8c94b80f4d3cc5ab4d0e22e8).
#
# This script, called by ../run.sh, creates the MUSAN
# data directory. The required dataset is freely available at
#   http://www.openslr.org/17/

if [ $# != 2 ]; then
  echo "Usage: $0 <path-to-dihard_2018_dev> <path-to-output>"
  echo " e.g.: $0 /export/corpora/LDC/LDC2018E31 data/dihard_2018_dev"
fi

path_to_dihard_2018_dev=$1
data_dir=$2

echo "Preparing ${data_dir}..."
local/make_dihard_2018_dev.py ${path_to_dihard_2018_dev} ${data_dir}

sort -k 2,2 -s ${data_dir}/rttm > ${data_dir}/rttm_tmp
mv ${data_dir}/rttm_tmp ${data_dir}/rttm
sort -k 1,1 -s ${data_dir}/reco2num_spk > ${data_dir}/reco2num_spk_tmp
mv ${data_dir}/reco2num_spk_tmp ${data_dir}/reco2num_spk
utils/fix_data_dir.sh ${data_dir}
