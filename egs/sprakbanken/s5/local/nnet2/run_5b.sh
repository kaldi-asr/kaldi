#!/bin/bash


stage=0
train_stage=-100
# This trains only unadapted (just cepstral mean normalized) features,
# and uses various combinations of VTLN warping factor and time-warping
# factor to artificially expand the amount of data.

. cmd.sh

. utils/parse_options.sh  # to parse the --stage option, if given

[ $# != 0 ] && echo "Usage: local/run_4b.sh [--stage <stage> --train-stage <train-stage>]" && exit 1;

set -e

if [ $stage -le 0 ]; then 
  # Create the training data.
  featdir=`pwd`/mfcc/nnet5b; mkdir -p $featdir
  fbank_conf=conf/fbank_40.conf
  echo "--num-mel-bins=40" > $fbank_conf
  steps/nnet2/get_perturbed_feats.sh --cmd "$train_cmd" \
    $fbank_conf $featdir exp/perturbed_fbanks_si284 data/train_si284 data/train_si284_perturbed_fbank &
  steps/nnet2/get_perturbed_feats.sh --cmd "$train_cmd" --feature-type mfcc \
    conf/mfcc.conf $featdir exp/perturbed_mfcc_si284 data/train_si284 data/train_si284_perturbed_mfcc &
  wait
fi

if [ $stage -le 1 ]; then
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train_si284_perturbed_mfcc data/lang exp/tri4b exp/tri4b_ali_si284_perturbed_mfcc
fi 

if [ $stage -le 2 ]; then
  steps/nnet2/train_block.sh --stage "$train_stage" \
     --cleanup false \
     --initial-learning-rate 0.01 --final-learning-rate 0.001 \
     --num-epochs 10 --num-epochs-extra 5 \
     --cmd "$decode_cmd" \
     --hidden-layer-dim 1536 \
     --num-block-layers 3 --num-normal-layers 3 \
      data/train_si284_perturbed_fbank data/lang exp/tri4b_ali_si284_perturbed_mfcc exp/nnet5b  || exit 1
fi

if [ $stage -le 3 ]; then # create testing fbank data.
  featdir=`pwd`/mfcc
  fbank_conf=conf/fbank_40.conf
  for x in test_eval92 test_eval93 test_dev93; do 
    cp -rT data/$x data/${x}_fbank
    rm -r ${x}_fbank/split* || true
    steps/make_fbank.sh --fbank-config "$fbank_conf" --nj 8 \
      --cmd "$train_cmd" data/${x}_fbank exp/make_fbank/$x $featdir  || exit 1;
    steps/compute_cmvn_stats.sh data/${x}_fbank exp/make_fbank/$x $featdir  || exit 1;
  done
fi

if [ $stage -le 4 ]; then
  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 \
     exp/tri4b/graph_bd_tgpr data/test_dev93_fbank exp/nnet5b/decode_bd_tgpr_dev93

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 8 \
     exp/tri4b/graph_bd_tgpr data/test_eval92_fbank exp/nnet5b/decode_bd_tgpr_eval92
fi



exit 0;

