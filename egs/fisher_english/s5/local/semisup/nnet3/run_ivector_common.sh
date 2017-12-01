#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script is similar to local/nnet3/run_ivector_common.sh, but 
# designed specifically for semi-supervised recipes. 
# This script accepts an optional argument of --unsup-train-set for
# unsupervised data directory, which will be combined with the supervised
# data directory to create semi-supervised data directory (whose name 
# is taken from the argument --semisup-train-set")

. ./cmd.sh
set -e
stage=-1
speed_perturb=true
train_set=train  # Supervised training set

# Unsupervised training set. 
# If provided, it will be combined with supervised training set to 
# create "semisup_train_set". This is the set that will be used to
# train the PCA transform and i-vector extractor.
unsup_train_set=  
semisup_train_set=

nnet3_affix=
exp=exp   # experiments directory. It could be something like exp/semisup_15k.

. ./path.sh
. ./utils/parse_options.sh

if [ ! -z "$unsup_train_set" ] && [ -z "$semisup_train_set" ]; then
  echo "$0: --semisup-train-set must be provided if --unsup-train-set is provided"
  exit 1
fi

if [ -z "$unsup_train_set" ] && [ ! -z "$semisup_train_set" ]; then
  echo "$0: --unsup-train-set must be provided if --semisup-train-set is provided"
  exit 1
fi

if [ ! -f data/$train_set/utt2spk ]; then
  echo "$0: data/$train_set/utt2spk does not exist"
  exit 1
fi

if [ ! -z "$unsup_train_set" ]; then
  if [ ! -f data/$unsup_train_set/utt2spk ]; then
    echo "$0: data/$unsup_train_set/utt2spk does not exist"
    exit 1
  fi

  # Combine supervised and unsupervised sets to create the 
  # semi-supervised training set.
  if [ $stage -le 0 ]; then
    utils/combine_data.sh data/$semisup_train_set \
      data/$train_set data/$unsup_train_set || exit 1
  fi
fi

# perturbed data preparation
if [ "$speed_perturb" == "true" ]; then
  if [ $stage -le 1 ]; then
    # Although the nnet will be trained by high resolution data, we still have
    # to perturb the normal data to get the alignments.
    # _sp stands for speed-perturbed

    for datadir in ${train_set} ${unsup_train_set}; do
      utils/data/perturb_data_dir_speed_3way.sh data/${datadir} data/${datadir}_sp
      utils/fix_data_dir.sh data/${datadir}_sp

      mfccdir=mfcc_perturbed
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
        data/${datadir}_sp exp/make_mfcc/${datadir}_sp $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/${datadir}_sp exp/make_mfcc/${datadir}_sp $mfccdir || exit 1;
      utils/fix_data_dir.sh data/${datadir}_sp
    done
  fi
fi

if [ ! -z "$unsup_train_set" ]; then
  if [ -f data/${semisup_train_set}_sp/feats.scp ]; then
    echo "$0: data/${semisup_train_set}_sp/feats.scp already exists! Remove it and try again."
    exit 1
  fi

  if [ $stage -le 2 ]; then
    utils/combine_data.sh data/${semisup_train_set}_sp \
      data/${train_set}_sp data/${unsup_train_set}_sp
  fi
fi

if [ $stage -le 3 ]; then
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english-$date/s5b/$mfccdir/storage $mfccdir/storage
  fi

  for dataset in $train_set $unsup_train_set; do
    utils/copy_data_dir.sh data/${dataset}_sp data/${dataset}_sp_hires
    utils/data/perturb_data_dir_volume.sh data/${dataset}_sp_hires

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/${dataset}_sp_hires exp/make_hires/${dataset}_sp $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_sp_hires exp/make_hires/${dataset}_sp $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_sp_hires;
  done

  for dataset in test dev; do
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 --mfcc-config conf/mfcc_hires.conf \
        data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires  # remove segments with problems
  done
fi

ivector_train_set=${train_set}_sp
if [ ! -z "$unsup_train_set" ]; then
  if [ -f data/${semisup_train_set}_sp_hires/feats.scp ]; then
    echo "$0: data/${semisup_train_set}_sp_hires/feats.scp already exists! Remove it and try again."
    exit 1
  fi

  if [ $stage -le 3 ]; then
    utils/combine_data.sh data/${semisup_train_set}_sp_hires \
      data/${train_set}_sp_hires data/${unsup_train_set}_sp_hires
  fi
  ivector_train_set=${semisup_train_set}_sp
fi

# ivector extractor training
if [ $stage -le 4 ]; then
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 10000 --subsample 2 \
    data/${ivector_train_set}_hires \
    $exp/nnet3${nnet3_affix}/pca_transform
fi

if [ $stage -le 5 ]; then
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 200000 \
    data/${ivector_train_set}_hires 512 \
    $exp/nnet3${nnet3_affix}/pca_transform $exp/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 6 ]; then
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/${ivector_train_set}_hires $exp/nnet3${nnet3_affix}/diag_ubm $exp/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 7 ]; then
  # We extract iVectors on all the ${train_set} data, which will be what we
  # train the system on.
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/${ivector_train_set}_hires data/${ivector_train_set}_max2_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/${ivector_train_set}_max2_hires $exp/nnet3${nnet3_affix}/extractor $exp/nnet3${nnet3_affix}/ivectors_${ivector_train_set}_hires || exit 1;
fi

if [ $stage -le 8 ]; then
  for dataset in test dev; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
      data/${dataset}_hires $exp/nnet3${nnet3_affix}/extractor $exp/nnet3${nnet3_affix}/ivectors_${dataset}_hires || exit 1;
  done
fi

exit 0;

