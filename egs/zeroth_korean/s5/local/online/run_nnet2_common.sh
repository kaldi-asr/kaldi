#!/bin/bash

# this script contains some common (shared) parts of the run_nnet*.sh scripts.
# Modified by Lucas Jo 2017 (Altas Guide)
. cmd.sh


stage=0

set -e
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if [ $stage -le 1 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=mfcc
  #if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
  #  utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/librispeech-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  #fi

  for datadir in train_2x; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
    steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;
  done

  # now create some data subsets.
  # mixed is the clean+other data.
  # a is 1/5 of the data, b is 2/5th of it.
  utils/subset_data_dir.sh data/train_2x_hires 3000 data/train_mixed_hires_a
  utils/subset_data_dir.sh data/train_2x_hires 6000 data/train_mixed_hires_b
fi

if [ $stage -le 2 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We align a subset of training data for
  # this purpose.
  utils/subset_data_dir.sh --utt-list <(awk '{print $1}' data/train_mixed_hires_a/utt2spk) \
     data/train_2x data/train_2x_a

  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
    data/train_2x_a data/lang exp/tri5b exp/nnet2_online/tri5b_ali_a
fi

if [ $stage -le 3 ]; then
  # Train a small system just for its LDA+MLLT transform.  We use --num-iters 13
  # because after we get the transform (12th iter is the last), any further
  # training is pointless.
    #5000 10000 data/train_mixed_hires_a data/lang \
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --realign-iters "" \
    --splice-opts "--left-context=3 --right-context=3" \
    3000 20000 data/train_mixed_hires_a data/lang \
    exp/nnet2_online/tri5b_ali_a exp/nnet2_online/tri6b
fi


if [ $stage -le 4 ]; then
  mkdir -p exp/nnet2_online
  # To train a diagonal UBM we don't need very much data, so use a small subset
  # (actually, it's not that small: still around 100 hours).
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 400000 \
    data/train_mixed_hires_a 256 exp/nnet2_online/tri6b exp/nnet2_online/diag_ubm
fi

if [ $stage -le 5 ]; then
  # iVector extractors can in general be sensitive to the amount of data, but
  # this one has a fairly small dim (defaults to 100) so we don't use all of it,
  # we use just the 3k subset (about one fifth of the data, or 200 hours).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/train_mixed_hires_b exp/nnet2_online/diag_ubm exp/nnet2_online/extractor || exit 1;
fi

if [ $stage -le 6 ]; then
  ivectordir=exp/nnet2_online/ivectors_train_2x_hires
  #if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
  #  utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/librispeech-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  #fi

  # We extract iVectors on all the train data, which will be what we train the
  # system on.  With --utts-per-spk-max 2, the script.  pairs the utterances
  # into twos, and treats each of these pairs as one speaker.  Note that these
  # are extracted 'online'.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/train_2x_hires data/train_2x_hires_max2
  
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 60 \
    data/train_2x_hires_max2 exp/nnet2_online/extractor $ivectordir || exit 1;
fi


exit 0;
