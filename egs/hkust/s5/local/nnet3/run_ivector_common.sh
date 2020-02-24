#!/usr/bin/env bash

# This script is modified based on swbd/s5c/local/nnet3/run_ivector_common.sh

# this script contains some common (shared) parts of the run_nnet*.sh scripts.

. ./cmd.sh


stage=0
num_threads_ubm=32
ivector_extractor=

set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

gmm_dir=exp/tri5a
align_script=steps/align_fmllr.sh

if [ $stage -le 1 ] && [ -z $ivector_extractor ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13) with pitch.
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/mfcc/hkust-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in train dev; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
    if [ "$datadir" == "train" ]; then
      dir=data/train_hires
      cat $dir/wav.scp | python -c "
import sys, os, subprocess, re, random
scale_low = 1.0/8
scale_high = 2.0
for line in sys.stdin.readlines():
  if len(line.strip()) == 0:
    continue
  print '{0} sox --vol {1} -t wav - -t wav - |'.format(line.strip(), random.uniform(scale_low, scale_high))
"| sort -k1,1 -u  > $dir/wav.scp_scaled || exit 1;
     mv $dir/wav.scp $dir/wav.scp_nonorm
     mv $dir/wav.scp_scaled $dir/wav.scp
    fi

    steps/make_mfcc_pitch_online.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;

    # make MFCC data dir without pitch to extract iVector
    utils/data/limit_feature_dim.sh 0:39 data/${datadir}_hires data/${datadir}_hires_nopitch || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires_nopitch exp/make_hires/$datadir $mfccdir || exit 1;
  done
fi

if [ $stage -le 2 ] && [ -z $ivector_extractor ]; then
  # perform PCA on the data
  echo "$0: computing a PCA transform from the no-pitch hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 10000 --subsample 2 \
    data/${train_set}_hires_nopitch \
    exp/nnet3/tri5_pca
fi

if [ $stage -le 3 ] && [ -z $ivector_extractor ]; then
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
    --num-frames 700000 \
    data/train_hires_nopitch 512 exp/nnet3/tri5_pca exp/nnet3/diag_ubm
fi

if [ $stage -le 4 ] && [ -z $ivector_extractor ]; then
  # iVector extractors can in general be sensitive to the amount of data, but
  # this one has a fairly small dim (defaults to 100)
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/train_hires_nopitch exp/nnet3/diag_ubm exp/nnet3/extractor || exit 1;
  ivector_extractor=exp/nnet3/extractor
fi

if [ $stage -le 5 ]; then
  # Although the nnet will be trained by high resolution data,
  # we still have to perturbe the normal data to get the alignment
  # _sp stands for speed-perturbed
  utils/perturb_data_dir_speed.sh 0.9 data/train data/temp1
  utils/perturb_data_dir_speed.sh 1.0 data/train data/temp2
  utils/perturb_data_dir_speed.sh 1.1 data/train data/temp3
  utils/combine_data.sh --extra-files utt2uniq data/train_sp data/temp1 data/temp2 data/temp3
  rm -r data/temp1 data/temp2 data/temp3

  mfccdir=mfcc_perturbed
  for x in train_sp; do
    steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" --nj 70 \
      data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  done
  utils/fix_data_dir.sh data/train_sp

  $align_script --nj 30 --cmd "$train_cmd" \
    data/train_sp data/lang $gmm_dir ${gmm_dir}_sp_ali || exit 1

  # Now perturb the high resolution data
  utils/copy_data_dir.sh data/train_sp data/train_sp_hires
  mfccdir=mfcc_perturbed_hires
  for x in train_sp_hires; do
    steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" --nj 70 --mfcc-config conf/mfcc_hires.conf \
      data/$x exp/make_hires/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_hires/$x $mfccdir || exit 1;
    # create MFCC data dir without pitch to extract iVector
    utils/data/limit_feature_dim.sh 0:39 data/$x data/${x}_nopitch || exit 1;
    steps/compute_cmvn_stats.sh data/${x}_nopitch exp/make_hires/$x $mfccdir || exit 1;
  done
  utils/fix_data_dir.sh data/train_sp_hires
fi

train_set=train_sp
if [ -z $ivector_extractor ]; then
  echo "iVector extractor is not found!"
  exit 1;
fi

if [ $stage -le 6 ]; then
  rm -f exp/nnet3/.error 2>/dev/null
  ivectordir=exp/nnet3/ivectors_${train_set}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/ivectors/hkust-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi
  # We extract iVectors on all the train data, which will be what we train the
  # system on.  With --utts-per-spk-max 2, the script.  pairs the utterances
  # into twos, and treats each of these pairs as one speaker.  Note that these
  # are extracted 'online'.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/${train_set}_hires_nopitch data/${train_set}_hires_nopitch_max2
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/${train_set}_hires_nopitch_max2 \
    $ivector_extractor $ivectordir \
    || touch exp/nnet3/.error
  [ -f exp/nnet3/.error ] && echo "$0: error extracting iVectors." && exit 1;
fi

if [ $stage -le 7 ]; then
  rm -f exp/nnet3/.error 2>/dev/null
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 8 \
    data/dev_hires_nopitch $ivector_extractor exp/nnet3/ivectors_dev || touch exp/nnet3/.error &
  wait
  [ -f exp/nnet3/.error ] && echo "$0: error extracting iVectors." && exit 1;
fi

exit 0;
