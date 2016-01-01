#!/bin/bash

# This script creates speed perturbed versions of the training data
# and generates the corresponding alignments

mic=ihm
nj=10
stage=0
use_sat_alignments=true

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

speed_perturb_datadir() {
  mic=$1
  dataset=$2
  extract_features=$3

  utils/perturb_data_dir_speed.sh 0.9 data/$mic/$dataset data/$mic/temp1
  utils/perturb_data_dir_speed.sh 1.0 data/$mic/$dataset data/$mic/temp2
  utils/perturb_data_dir_speed.sh 1.1 data/$mic/$dataset data/$mic/temp3
  utils/combine_data.sh --extra-files utt2uniq data/$mic/${dataset}_sp data/$mic/temp1 data/$mic/temp2 data/$mic/temp3
  rm -r data/$mic/temp1 data/$mic/temp2 data/$mic/temp3

  if [ "$extract_features" == "true" ]; then
    mfccdir=mfcc_${mic}_perturbed
    for x in ${dataset}_sp; do
      steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
        data/$mic/$x exp/make_${mic}_mfcc/$x $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/$mic/$x exp/make_${mic}_mfcc/$x $mfccdir || exit 1;
    done
  fi
  utils/fix_data_dir.sh data/$mic/${dataset}_sp
}


if [ $stage -le 1 ]; then
  #Although the nnet will be trained by high resolution data, we still have to perturb the normal data to get the alignment
  # _sp stands for speed-perturbed
    speed_perturb_datadir $mic train true
fi


if [ $stage -le 2 ]; then
  # we just need to recreate alignments in case we perturbed the data 
  # or in the case we are using ihm alignments, else the alignments would already
  # have been generated when we built the GMM-HMM systems
  data_set=train_sp
  if [ "$use_sat_alignments" == "true" ]; then
    gmm_dir=exp/$mic/tri4a
    align_script=steps/align_fmllr.sh
  else
    gmm_dir=exp/$mic/tri3a
    align_script=steps/align_si.sh
  fi
  $align_script --nj $nj --cmd "$train_cmd" \
    data/$mic/train_sp data/lang $gmm_dir ${gmm_dir}_${mic}_${data_set}_ali || exit 1;
fi

exit 0;
