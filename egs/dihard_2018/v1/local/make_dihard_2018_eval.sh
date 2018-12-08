#!/bin/bash
# Copyright 2018   Zili Huang
# Apache 2.0.
#
# This script, called by ../run.sh, creates the DIHARD 2018 evaluation directory.

if [ $# != 2 ]; then
  echo "Usage: $0 <path-to-dihard_2018_eval> <path-to-output>"
  echo " e.g.: $0 /export/corpora/LDC/LDC2018E32v1.1 data/dihard_2018_eval"
fi

path_to_dihard_2018_eval=$1
data_dir=$2

echo "Preparing ${data_dir}..."
local/make_dihard_2018_eval.py ${path_to_dihard_2018_eval} ${data_dir}

sort -k 2,2 -s ${data_dir}/rttm > ${data_dir}/rttm_tmp
mv ${data_dir}/rttm_tmp ${data_dir}/rttm
sort -k 1,1 -s ${data_dir}/reco2num_spk > ${data_dir}/reco2num_spk_tmp
mv ${data_dir}/reco2num_spk_tmp ${data_dir}/reco2num_spk
utils/fix_data_dir.sh ${data_dir}
