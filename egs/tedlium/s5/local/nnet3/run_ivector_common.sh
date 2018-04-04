#!/bin/bash

# This is based on:
# swbd/s5c/local/nnet3/run_ivector_common.sh and
# tedlium/s5/local/online/run_nnet2_ms_perturbed.sh
# see the chain docs for general direction on what training is doing!

set -uo pipefail
stage=1
generate_alignments=true # false if doing ctc training

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

mkdir -p exp/nnet3 
# perturb the data
train_set=train
if [ $stage -le 1 ]; then
  #Although the nnet will be trained by high resolution data, we still have to perturb the normal data to get the alignment

  utils/perturb_data_dir_speed.sh 0.9 data/${train_set} data/temp1
  utils/perturb_data_dir_speed.sh 1.1 data/${train_set} data/temp2
  utils/combine_data.sh data/${train_set}_tmp data/temp1 data/temp2
  utils/validate_data_dir.sh --no-feats data/${train_set}_tmp
  rm -r data/temp1 data/temp2

  mfccdir=mfcc_perturbed
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
    data/${train_set}_tmp exp/make_mfcc/${train_set}_tmp $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_tmp exp/make_mfcc/${train_set}_tmp $mfccdir || exit1;
  utils/fix_data_dir.sh data/${train_set}_tmp
    
  utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- data/${train_set} data/temp0
  utils/combine_data.sh data/${train_set}_sp data/${train_set}_tmp data/temp0
  utils/fix_data_dir.sh data/${train_set}_sp
  rm -r data/temp0 data/${train_set}_tmp
fi

train_set_sp=${train_set}_sp

if [ $stage -le 2 ] && [ "$generate_alignments" == "true" ]; then
  # obtain the alignment of the pertubed data
  steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
    data/${train_set_sp} data/lang_nosp exp/tri3 exp/tri3_ali_sp || exit 1
fi

if [ $stage -le 3 ]; then
  
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b{09,10,11,12}/$USER/kaldi-data/egs/tedlium-$date/s5/$mfccdir/storage $mfccdir/storage
  fi

  for dataset in $train_set $train_set_sp; do
    data_dir=data/${dataset}_hires
    utils/copy_data_dir.sh data/$dataset $data_dir

      # this next section does volume perturbation on the data.
    cat $data_dir/wav.scp | python -c "
import sys, os, subprocess, re, random
random.seed(0)
scale_low = 1.0/8
scale_high = 2.0
for line in sys.stdin.readlines():
  if len(line.strip()) == 0:
    continue
  print '{0} sox --vol {1} -t wav - -t wav - |'.format(line.strip(), random.uniform(scale_low, scale_high))
"| sort -k1,1 -u  > $data_dir/wav.scp_scaled || exit 1;
    mv $data_dir/wav.scp_scaled $data_dir/wav.scp

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
      $data_dir exp/make_hires/$dataset $mfccdir
    steps/compute_cmvn_stats.sh $data_dir exp/make_hires/$dataset $mfccdir
    utils/fix_data_dir.sh $data_dir # remove segments with problems
  done

  for dataset in dev test; do
    data_dir=data/${dataset}_hires
    utils/copy_data_dir.sh data/$dataset $data_dir

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
      $data_dir exp/make_hires/$dataset $mfccdir
    steps/compute_cmvn_stats.sh $data_dir exp/make_hires/$dataset $mfccdir
    utils/fix_data_dir.sh $data_dir # remove segments with problems
  done
fi

# ivector extractor training
if [ $stage -le 5 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
  # the transform (12th iter is the last), any further training is pointless.
  # this decision is based on fisher_english
  # Note: We do NOT use speed-perturbed data in this step.
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 10000 data/${train_set}_hires \
    data/lang_nosp exp/tri3_ali exp/nnet3/tri3b
fi

if [ $stage -le 6 ]; then
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 700000 \
    data/${train_set_sp}_hires 512 exp/nnet3/tri3b exp/nnet3/diag_ubm
fi

if [ $stage -le 7 ]; then
    steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
        data/${train_set_sp}_hires exp/nnet3/diag_ubm exp/nnet3/extractor || exit 1;
fi

if [ $stage -le 8 ]; then
  
    steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/${train_set_sp}_hires \
        data/${train_set_sp}_hires_max2

    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
        data/${train_set_sp}_hires_max2 exp/nnet3/extractor exp/nnet3/ivectors_${train_set_sp} || exit 1

    for data_set in dev test; do
      steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 8 \
        data/${data_set}_hires exp/nnet3/extractor exp/nnet3/ivectors_${data_set} || exit 1;
    done
fi
