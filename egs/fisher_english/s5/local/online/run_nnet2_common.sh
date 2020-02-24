#!/usr/bin/env bash

# Make the features, build the iVector extractor


. ./cmd.sh

stage=1
set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

mkdir -p exp/nnet2_online

if [ $stage -le 1 ]; then
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=mfcc
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english-$date/s5/$mfccdir/storage $mfccdir/storage
  fi
  utils/copy_data_dir.sh data/train data/train_hires
  steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/train_hires exp/make_hires/train $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/train_hires exp/make_hires/train $mfccdir || exit 1;

  utils/subset_data_dir.sh data/train_hires 30000 data/train_hires_30k
  # want the 100k subset to exactly match train_100k, since we'll use its alignments.
  awk '{print $1}' data/train_100k/utt2spk > uttlist
  utils/subset_data_dir.sh --utt-list uttlist data/train_hires data/train_hires_100k
  rm uttlist
fi

if [ $stage -le 2 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
  # the transform (12th iter is the last), any further training is pointless.
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 10000 data/train_hires_100k data/lang exp/tri4a exp/nnet2_online/tri5a
fi


if [ $stage -le 3 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest
  # subset.  the input directory exp/nnet2_online/tri5a is only needed for
  # the splice-opts and the LDA transform.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 400000 \
    data/train_hires_30k 512 exp/nnet2_online/tri5a exp/nnet2_online/diag_ubm
fi

if [ $stage -le 4 ]; then
  # iVector extractors can in general be sensitive to the amount of data, but
  # this one has a fairly small dim (defaults to 100) so we don't use all of it,
  # we use just the 100k subset (about one sixteenth of the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/train_hires_100k exp/nnet2_online/diag_ubm exp/nnet2_online/extractor || exit 1;
fi

if [ $stage -le 5 ]; then
  ivectordir=exp/nnet2_online/ivectors_train
  if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then # this shows how you can split across multiple file-systems.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english/s5/$ivectordir/storage $ivectordir/storage
  fi

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/train_hires data/train_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 60 \
    data/train_hires_max2 exp/nnet2_online/extractor $ivectordir || exit 1;
fi
