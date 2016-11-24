#!/bin/bash 
#set -e
# this script is based on local/nnet3/run_ivector_common.sh
# but it operates on corrupted training/dev/test data sets

. cmd.sh

stage=1
snrs="20:10:15:5:0"
num_data_reps=3
ali_dir=exp/tri4_ali_nodup_sp

set -e
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

mkdir -p exp/nnet3_reverberate
if [ $stage -le 0 ]; then
  # prepare the impulse responses
  local/reverberate/prep_sim_rirs.sh \
    --num_room 50 \
    --rir_per_room 100 \
    data/impulses_noises_sim
fi
    

if [ $stage -le 1 ]; then
  # corrupt the data to generate reverberated data 
  for data_dir in train_nodup_sp eval2000 train_dev ; do
    if [ "$data_dir" == "train_nodup_sp" ]; then
      num_reps=$num_data_reps
    else
      num_reps=1
    fi
    python local/reverberate/reverberate_data_dir.py \
      --rir-list data/impulses_noises_sim/rir_list \
      --num-replications $num_data_reps \
      data/${data_dir} data/${data_dir}_rvb
  done

fi

