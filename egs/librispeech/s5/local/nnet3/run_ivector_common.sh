#!/bin/bash

# this script contains some common (shared) parts of the run_nnet*.sh scripts.

. cmd.sh


stage=0
generate_alignments=true # false if doing ctc training
speed_perturb=true

set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


train_set=train_960
if [ "$speed_perturb" == "true" ]; then
  if [ $stage -le 1 ]; then
    #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
    # _sp stands for speed-perturbed

    for datadir in train_960; do
      utils/perturb_data_dir_speed.sh 0.9 data/${datadir} data/temp1
      utils/perturb_data_dir_speed.sh 1.1 data/${datadir} data/temp2
      utils/combine_data.sh data/${datadir}_tmp data/temp1 data/temp2
      utils/validate_data_dir.sh --no-feats data/${datadir}_tmp
      rm -r data/temp1 data/temp2

      mfccdir=mfcc_perturbed
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
        data/${datadir}_tmp exp/make_mfcc/${datadir}_tmp $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/${datadir}_tmp exp/make_mfcc/${datadir}_tmp $mfccdir || exit 1;
      utils/fix_data_dir.sh data/${datadir}_tmp

      utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- data/${datadir} data/temp0
      utils/combine_data.sh data/${datadir}_sp data/${datadir}_tmp data/temp0
      utils/fix_data_dir.sh data/${datadir}_sp
      rm -r data/temp0 data/${datadir}_tmp
    done
  fi

  if [ $stage -le 2 ] && [ "$generate_alignments" == "true" ]; then
    #obtain the alignment of the perturbed data
    steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
      data/train_960_sp data/lang exp/tri6b exp/tri6b_sp || exit 1
  fi
  train_set=train_960_sp
fi

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/librispeech-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in $train_set test_clean test_other dev_clean dev_other; do
    if [ "$datadir" == "$train_set" ]; then
      utils/data/perturb_data_dir_volume.sh data/$datadir
    fi
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;
  done

  # now create some data subsets.
  # mixed is the clean+other data.
  # 30k is 1/10 of the data (around 100 hours), 60k is 1/5th of it (around 200 hours).
  utils/subset_data_dir.sh data/${train_set}_hires 30000 data/${train_set}_mixed_hires_30k
  utils/subset_data_dir.sh data/${train_set}_hires 60000 data/${train_set}_mixed_hires_60k
fi

if [ $stage -le 4 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We align a subset of training data for
  # this purpose.
  utils/subset_data_dir.sh --utt-list <(awk '{print $1}' data/${train_set}_mixed_hires_30k/utt2spk) \
     data/${train_set} data/${train_set}_30k

  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
    data/${train_set}_30k data/lang exp/tri6b exp/nnet3/tri6b_ali_30k
fi

if [ $stage -le 5 ]; then
  # Train a small system just for its LDA+MLLT transform.  We use --num-iters 13
  # because after we get the transform (12th iter is the last), any further
  # training is pointless.
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --realign-iters "" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 10000 data/${train_set}_mixed_hires_30k data/lang \
    exp/nnet3/tri6b_ali_30k exp/nnet3/tri7b
fi


if [ $stage -le 6 ]; then
  mkdir -p exp/nnet3
  # To train a diagonal UBM we don't need very much data, so use a small subset
  # (actually, it's not that small: still around 100 hours).
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 700000 \
    data/${train_set}_mixed_hires_30k 512 exp/nnet3/tri7b exp/nnet3/diag_ubm
fi

if [ $stage -le 7 ]; then
  # iVector extractors can in general be sensitive to the amount of data, but
  # this one has a fairly small dim (defaults to 100) so we don't use all of it,
  # we use just the 60k subset (about one fifth of the data, or 200 hours).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/${train_set}_mixed_hires_60k exp/nnet3/diag_ubm exp/nnet3/extractor || exit 1;
fi

if [ $stage -le 8 ]; then
  ivectordir=exp/nnet3/ivectors_${train_set}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/b{09,10,11,12}/$USER/kaldi-data/egs/librispeech-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi
  # We extract iVectors on all the train data, which will be what we train the
  # system on.  With --utts-per-spk-max 2, the script.  pairs the utterances
  # into twos, and treats each of these pairs as one speaker.  Note that these
  # are extracted 'online'.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/${train_set}_hires data/${train_set}_hires_max2
  
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 60 \
    data/${train_set}_hires_max2 exp/nnet3/extractor $ivectordir || exit 1;
fi

if [ $stage -le 9 ]; then
  for data in test_clean test_other dev_clean dev_other; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
      data/${data}_hires exp/nnet3/extractor exp/nnet3/ivectors_${data} || exit 1;
   done
   wait
fi

exit 0;
