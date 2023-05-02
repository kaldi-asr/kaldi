#!/usr/bin/env bash

# This script creates the parallel data dir based on ihm data,
# creates speed perturbed versions of this parallel data
# and generates the corresponding alignments.
# The parallel data dir has segment ids from distant microphone data
# but the wav data is copied from ihm.

mic=sdm1
new_mic=sdm1_cleanali
use_sat_alignments=true
nj=10
stage=0

. ./cmd.sh
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
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
      utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/ami-$mic-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
    fi
    for x in ${dataset}_sp; do
      steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
        data/$mic/$x exp/make_${mic}_mfcc/$x $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/$mic/$x exp/make_${mic}_mfcc/$x $mfccdir || exit 1;
    done
  fi
  utils/fix_data_dir.sh data/$mic/${dataset}_sp
}

if [ $stage -le 0 ]; then
  # we will use ihm alignments as targets
  # but as the segment names differ we will create a new data dir 
  local/nnet3/prepare_parallel_datadirs.sh --original-mic $mic \
                                           --parallel-mic ihm \
                                           --new-mic $new_mic
fi

mic=$new_mic
if [ $stage -le 1 ]; then
# extract the features for the parallel data dir which will be used for alignments
# in case there is no speed perturbation
  mfccdir=mfcc_${mic}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/ami-$mic-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
    data/${mic}/train_parallel exp/make_${mic}_mfcc/train_parallel $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$mic/train_parallel exp/make_${mic}_mfcc/train_parallel $mfccdir || exit 1;
  utils/fix_data_dir.sh data/$mic/train_parallel
fi

if [ $stage -le 2 ]; then
  # if we are using the ihm alignments we just need features for the parallel
  # data, the actual data is being perturbed just so that we can copy this 
  # directory to create hiresolution features later
  speed_perturb_datadir $mic train_parallel true 
  speed_perturb_datadir $mic train false
fi

if [ $stage -le 3 ]; then
  # we just need to recreate alignments in case we perturbed the data 
  # or in the case we are using ihm alignments, else the alignments would already
  # have been generated when we built the GMM-HMM systems
  data_set=train_parallel_sp
  if [ "$use_sat_alignments" == "true" ]; then
    gmm_dir=exp/ihm/tri4a
    align_script=steps/align_fmllr.sh
  else
    gmm_dir=exp/ihm/tri3a
    align_script=steps/align_si.sh
  fi
  $align_script --nj $nj --cmd "$train_cmd" \
    data/$mic/train_parallel_sp data/lang $gmm_dir ${gmm_dir}_${mic}_${data_set}_ali || exit 1;
fi

exit 0;
