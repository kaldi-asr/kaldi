#!/bin/bash

# this script is called from scripts like run_ms.sh; it does the common stages
# of the build, such as feature extraction.
# This is actually the same as local/online/run_nnet2_common.sh, except
# for the directory names.

. cmd.sh
mfccdir=mfcc
mfcc_config=conf/mfcc_hires.conf
stage=1
suffix=_hires
extractor=exp/nnet3/extractor

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if [ $stage -le 1 ]; then
  for datadir in train_si284 test_eval93 test_dev93 test_eval92; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}${suffix}
    steps/make_mfcc.sh --nj 40 --mfcc-config $mfcc_config \
      --cmd "$train_cmd" data/${datadir}${suffix} exp/make_hires/$datadir $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}${suffix} exp/make_hires/$datadir $mfccdir || exit 1;
  done
  utils/subset_data_dir.sh --first data/train_si284${suffix} 7138 data/train_si84${suffix} || exit 1
fi

if [ $stage -le 2 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We align the si84 data for this purpose.

  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
    data/train_si84 data/lang exp/tri4b exp/nnet3/tri4b_ali_si84
fi

if [ ! -f $extractor/final.dubm ]; then
  echo hi
  extractor=exp/nnet3/extractor
  if [ $stage -le 3 ]; then
    # Train a small system just for its LDA+MLLT transform.  We use --num-iters 13
    # because after we get the transform (12th iter is the last), any further
    # training is pointless.
    steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
      --realign-iters "" \
      --splice-opts "--left-context=3 --right-context=3" \
      5000 10000 data/train_si84${suffix} data/lang \
       exp/nnet3/tri4b_ali_si84 exp/nnet3/tri5b
  fi

  if [ $stage -le 4 ]; then
    mkdir -p exp/nnet3

    steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
       --num-frames 400000 data/train_si84${suffix} 256 exp/nnet3/tri5b exp/nnet3/diag_ubm
  fi

  if [ $stage -le 5 ]; then
    # even though $nj is just 10, each job uses multiple processes and threads.
    steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
      data/train_si284${suffix} exp/nnet3/diag_ubm $extractor || exit 1;
  fi
fi

if [ $stage -le 6 ]; then
  # We extract iVectors on all the train_si284 data, which will be what we
  # train the system on.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/train_si284${suffix} \
    data/train_si284${suffix}_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/train_si284${suffix}_max2 $extractor exp/nnet3/ivectors_train_si284 || exit 1;
fi

if [ $stage -le 7 ]; then
  rm exp/nnet3/.error 2>/dev/null
  for data in test_eval92 test_dev93 test_eval93; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 8 \
      data/${data}${suffix} $extractor exp/nnet3/ivectors_${data} || touch exp/nnet3/.error &
  done
  wait
  [ -f exp/nnet3/.error ] && echo "$0: error extracting iVectors." && exit 1;
fi

exit 0;
