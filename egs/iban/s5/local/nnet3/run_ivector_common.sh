#!/bin/bash

## Script was adapted from WSJ (login) and RM (some settings)

. cmd.sh
mfccdir=mfcc

stage=1

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if [ $stage -le 1 ]; then
    for datadir in train; do
      utils/perturb_data_dir_speed.sh 0.9 data/${datadir} data/temp1
      utils/perturb_data_dir_speed.sh 1.1 data/${datadir} data/temp2
      utils/combine_data.sh data/${datadir}_tmp data/temp1 data/temp2
      utils/validate_data_dir.sh --no-feats data/${datadir}_tmp
      rm -r data/temp1 data/temp2

      mfccdir=mfcc_perturbed
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 17 \
        data/${datadir}_tmp exp/make_mfcc/${datadir}_tmp $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/${datadir}_tmp exp/make_mfcc/${datadir}_tmp $mfccdir || exit 1;
      utils/fix_data_dir.sh data/${datadir}_tmp

      utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- data/${datadir} data/temp0
      utils/combine_data.sh data/${datadir}_sp data/${datadir}_tmp data/temp0
      utils/fix_data_dir.sh data/${datadir}_sp
      rm -r data/temp0 data/${datadir}_tmp
    done
fi

mkdir -p exp/nnet3

if [ $stage -le 2 ]; then
    steps/align_fmllr.sh --nj 16 --cmd "$train_cmd" \
      data/train_sp data/lang exp/tri3b exp/nnet3/tri3b_ali_sp || exit 1
fi

mfccdir=mfcc_hires
if [ $stage -le 3 ]; then
   utils/copy_data_dir.sh data/train_sp data/train_hires || exit 1
   steps/make_mfcc.sh --nj 16 --mfcc-config conf/mfcc_hires.conf \
     --cmd "$train_cmd" data/train_hires exp/make_hires/train $mfccdir || exit 1;
   steps/compute_cmvn_stats.sh data/train_hires exp/make_hires/train $mfccdir || exit 1;

   for datadir in  dev; do
     utils/copy_data_dir.sh data/$datadir data/${datadir}_hires || exit 1
     steps/make_mfcc.sh --nj 6 --mfcc-config conf/mfcc_hires.conf \
       --cmd "$train_cmd" data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;
     steps/compute_cmvn_stats.sh data/${datadir}_hires exp/make_hires/$datadir $mfccdir || exit 1;
  done
fi

if [ $stage -le 4 ]; then
  # Train a small system just for its LDA+MLLT transform.  We use --num-iters 13
  # because after we get the transform (12th iter is the last), any further
  # training is pointless.
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --realign-iters "" --splice-opts "--left-context=3 --right-context=3" \
    5000 10000 data/train_hires data/lang \
     exp/nnet3/tri3b_ali_sp exp/nnet3/tri5b || exit 1
fi

if [ $stage -le 5 ]; then
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 16  --num-frames 200000 \
     data/train_hires 256 exp/nnet3/tri5b exp/nnet3/diag_ubm || exit 1
fi

if [ $stage -le 6 ]; then
  # even though $nj is just 10, each job uses multiple processes and threads.
 steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" \
    --nj 10 --num-processes 1 --num-threads 2  --ivector-dim 50\
    data/train_hires exp/nnet3/diag_ubm exp/nnet3/extractor || exit 1;
fi

if [ $stage -le 7 ]; then
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/train_hires \
    data/train_hires_max2 || exit 1

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 16\
    data/train_hires_max2 exp/nnet3/extractor exp/nnet3/ivectors_train || exit 1
fi

if [ $stage -le 8 ]; then
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 6 \
      data/dev_hires exp/nnet3/extractor exp/nnet3/ivectors_dev || exit 1
fi

exit 0;
