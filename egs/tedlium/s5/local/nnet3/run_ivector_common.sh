#!/bin/bash

# This is based on:
# swbd/s5c/local/nnet3/run_ivector_common.sh and
# tedlium/s5/local/online/run_nnet2_ms_perturbed.sh
# see the chain docs for general direction on what training is doing!

set -uo pipefail
stage=1
train_stage=-10
generate_alignments=true # false if doing ctc training
speed_perturb=true

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

mkdir -p exp/nnet3 
# perturbed data
train_set=train
if [ "$speed_perturb" == "true" ]; then
  if [ $stage -le 1 ]; then
  #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
  # _sp stands for speed-perturbed

  for datadir in $train_set; do
    utils/perturb_data_dir_speed.sh 0.9 data/${datadir} data/temp1
    utils/perturb_data_dir_speed.sh 1.1 data/${datadir} data/temp2
    utils/combine_data.sh data/${datadir}_tmp data/temp1 data/temp2
    utils/validate_data_dir.sh --no-feats data/${datadir}_tmp
    rm -r data/temp1 data/temp2

    mfccdir=mfcc_perturbed
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
      data/${datadir}_tmp exp/make_mfcc/${datadir}_tmp $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_tmp exp/make_mfcc/${datadir}_tmp $mfccdir || exit1;
    utils/fix_data_dir.sh data/${datadir}_tmp
    
    utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- data/${datadir} data/temp0
    utils/combine_data.sh data/${datadir}_sp data/${datadir}_tmp data/temp0
    utils/fix_data_dir.sh data/${datadir}_sp
    rm -r data/temp0 data/${datadir}_tmp
  done
  fi

  if [ $stage -le 2 ] && [ "$generate_alignments" == "true" ]; then
    # obtain the alignment of the pertubed data
    steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
      data/${train_set}_sp data/lang_nosp exp/tri3 exp/tri3_ali_sp || exit 1
  fi

  train_set=${train_set}_sp
fi

if [ $stage -le 3 ]; then
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b{09,10,11,12}/$USER/kaldi-data/egs/tedlium-$date/s5b/$mfccdir/storage $mfccdir/storage
  fi

  for dataset in $train_set dev test; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires

    data_dir=data/${dataset}_hires
      # this next section does volume perturbation on the data.
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

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
      data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires # remove segments with problems
  done
fi

if [ $stage -le 5 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 10000 data/${train_set}_hires \
    data/lang_nosp exp/tri3_ali_sp exp/nnet3/tri4
fi

if [ $stage -le 6 ]; then
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 700000 \
    data/${train_set}_hires 512 exp/nnet3/tri4 exp/nnet3/diag_ubm
fi

if [ $stage -le 7 ]; then
    steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
        data/${train_set}_hires exp/nnet3/diag_ubm exp/nnet3/extractor || exit 1;
fi

if [ $stage -le 8 ]; then
    steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/${train_set}_hires \
        data/${train_set}_hires_max2

    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
        data/${train_set}_hires_max2 exp/nnet3/extractor exp/nnet3/ivectors_${train_set} || exit 1;

    for data_set in dev test; do
      steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 8 \
        data/${data_set}_hires exp/nnet3/extractor exp/nnet3/ivectors_$data_set || exit 1;
    done
fi
