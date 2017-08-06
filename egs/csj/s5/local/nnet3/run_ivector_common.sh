#!/bin/bash

# This is modified from swbd/s5c/local/nnet3/run_ivector_common.sh
# Tomohiro Tanaka, 15/05/2016

# this script is called from scripts like run_ms.sh; it does the common stages
# of the build, such as feature extraction.
# This is actually the same as local/online/run_nnet2_common.sh, except
# for the directory names.

. ./cmd.sh
mfccdir=mfcc
stage=4

if [ -e data/train_dev ] ;then
    dev_set=train_dev
fi

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $stage -le 1 ]; then
    for datadir in train_nodup $dev_set eval1 eval2 eval3 ; do
	utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
	steps/make_mfcc.sh --nj 10 --mfcc-config conf/mfcc_hires.conf \
	    --cmd "$train_cmd" data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;
	steps/compute_cmvn_stats.sh data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;
    done
fi

if [ $stage -le 2 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We align the si84 data for this purpose.
  steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" \
    data/train_nodup data/lang exp/tri4 exp/nnet3/tri4_ali_train_nodup
fi

if [ $stage -le 3 ]; then
  # Train a small system just for its LDA+MLLT transform.  We use --num-iters 13
  # because after we get the transform (12th iter is the last), any further
  # training is pointless.
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --realign-iters "" \
    --splice-opts "--left-context=4 --right-context=4" \
    6000 140000 data/train_nodup_hires data/lang \
    exp/nnet3/tri4_ali_train_nodup exp/nnet3/tri5b
fi

if [ $stage -le 4 ]; then
  mkdir -p exp/nnet3
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 10 \
     --num-frames 500000 data/train_nodup_hires 512 exp/nnet3/tri5b exp/nnet3/diag_ubm
fi

if [ $stage -le 5 ]; then
  # even though $nj is just 10, each job uses multiple processes and threads.
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/train_nodup_hires exp/nnet3/diag_ubm exp/nnet3/extractor || exit 1;
fi

if [ $stage -le 6 ]; then
  # We extract iVectors on all the train_si284 data, which will be what we
  # train the system on.
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/train_nodup_hires \
    data/train_nodup_hires_max2
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 10 \
    data/train_nodup_hires_max2 exp/nnet3/extractor exp/nnet3/ivectors_train_nodup || exit 1;
fi

if [ $stage -le 7 ]; then
  rm exp/nnet3/.error 2>/dev/null
  for data in $dev_set eval1 eval2 eval3; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 10 \
      data/${data}_hires exp/nnet3/extractor exp/nnet3/ivectors_${data} || touch exp/nnet3/.error 
  done
  wait
  [ -f exp/nnet3/.error ] && echo "$0: error extracting iVectors." && exit 1;
fi

exit 0;