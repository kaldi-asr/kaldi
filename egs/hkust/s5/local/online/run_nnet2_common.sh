#!/bin/bash

. ./cmd.sh
set -e
stage=1
train_stage=-10

. ./path.sh
. ./utils/parse_options.sh

mkdir -p exp/nnet2_online

if [ $stage -le 1 ]; then
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/egs/hkust-$date/s5b/$mfccdir/storage $mfccdir/storage
  fi
  utils/copy_data_dir.sh data/train data/train_scaled_hires
  utils/copy_data_dir.sh data/train data/train_hires

  data_dir=data/train_scaled_hires
  cat $data_dir/wav.scp | python -c "
import sys, os, subprocess, re, random
scale_low = 1.0/8
scale_high = 2.0
for line in sys.stdin.readlines():
  if len(line.strip()) == 0:
    continue
  print '{0} sox --vol {1} -t wav - -t wav - |'.format(line.strip(), random.uniform(scale_low, scale_high))
"| sort -k1,1 -u  > $data_dir/wav.scp_scaled || exit 1;
  mv $data_dir/wav.scp_scaled $data_dir/wav.scp
  steps/make_mfcc_pitch_online.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/train_scaled_hires exp/make_hires/train_scaled $mfccdir;
  steps/compute_cmvn_stats.sh data/train_scaled_hires exp/make_hires/train_scaled $mfccdir;

  # we need these features for the run_nnet2_ms.sh
  steps/make_mfcc_pitch_online.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/train_hires exp/make_hires/train $mfccdir;
  steps/compute_cmvn_stats.sh data/train_hires exp/make_hires/train $mfccdir;

  # Remove the small number of utterances that couldn't be extracted for some
  # reason (e.g. too short; no such file).
  utils/fix_data_dir.sh data/train_scaled_hires;
  utils/fix_data_dir.sh data/train_hires;

  # Create MFCC+pitchs for the dev set
  utils/copy_data_dir.sh data/dev data/dev_hires
  steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" --nj 10 --mfcc-config conf/mfcc_hires.conf \
      data/dev_hires exp/make_hires/dev $mfccdir;
  steps/compute_cmvn_stats.sh data/dev_hires exp/make_hires/dev $mfccdir;
  utils/fix_data_dir.sh data/dev_hires  # remove segments with problems

  # Take the MFCCs for training iVector extractors
  utils/data/limit_feature_dim.sh 0:39 data/train_scaled_hires data/train_scaled_hires_nopitch || exit 1;
  steps/compute_cmvn_stats.sh data/train_scaled_hires_nopitch exp/make_hires/train $mfccdir || exit 1;
  utils/data/limit_feature_dim.sh 0:39 data/train_hires data/train_hires_nopitch || exit 1;
  steps/compute_cmvn_stats.sh data/train_hires_nopitch exp/make_hires/train $mfccdir || exit 1;
  utils/data/limit_feature_dim.sh 0:39 data/dev_hires data/dev_hires_nopitch || exit 1;
  steps/compute_cmvn_stats.sh data/dev_hires_nopitch exp/make_hires/dev $mfccdir || exit 1;

  # Take the first 30k utterances (about 1/5th of the data) this will be used
  # for the diagubm training
  utils/subset_data_dir.sh --first data/train_scaled_hires_nopitch 30000 data/train_scaled_hires_30k

  # create a 100k subset for the lda+mllt training
  utils/subset_data_dir.sh --first data/train_scaled_hires_nopitch 100000 data/train_scaled_hires_100k;
fi

if [ $stage -le 2 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
  # the transform (12th iter is the last), any further training is pointless.
  # this decision is based on fisher_english
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    5500 90000 data/train_scaled_hires_100k \
    data/lang exp/tri2_ali_100k exp/nnet2_online/tri3b
fi

if [ $stage -le 3 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 200000 \
    data/train_scaled_hires_30k 512 exp/nnet2_online/tri3b exp/nnet2_online/diag_ubm
fi

if [ $stage -le 4 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/train_scaled_hires_100k exp/nnet2_online/diag_ubm exp/nnet2_online/extractor || exit 1;
fi

if [ $stage -le 5 ]; then
  # We extract iVectors on all the train_nodup data, which will be what we
  # train the system on.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/train_hires_nopitch data/train_hires_nopitch_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/train_hires_nopitch_max2 exp/nnet2_online/extractor exp/nnet2_online/ivectors_train || exit 1;
fi


exit 0;
