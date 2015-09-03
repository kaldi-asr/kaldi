#!/bin/bash


stage=0
train_stage=-100
# This trains only unadapted (just cepstral mean normalized) features,
# and uses various combinations of VTLN warping factor and time-warping
# factor to artificially expand the amount of data.


. ./cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF

parallel_opts="-l gpu=1"  # This is suitable for the CLSP network, you'll likely have to change it.

. utils/parse_options.sh  # to parse the --stage option, if given

[ $# != 0 ] && echo "Usage: local/run_4b.sh [--stage <stage> --train-stage <train-stage>]" && exit 1;


set -e

if [ $stage -le 0 ]; then
  # Create the training data.                                                                                                                   
  featdir=`pwd`/mfcc/nnet4b; mkdir -p $featdir
  fbank_conf=conf/fbank_40.conf
  echo "--num-mel-bins=40" > $fbank_conf
  steps/nnet2/get_perturbed_feats.sh --cmd "$train_cmd" \
    $fbank_conf $featdir exp/perturbed_fbanks data/train data/train_perturbed_fbank &
  steps/nnet2/get_perturbed_feats.sh --cmd "$train_cmd" --feature-type mfcc \
    conf/mfcc.conf $featdir exp/perturbed_mfcc data/train data/train_perturbed_mfcc &
  wait
fi

if [ $stage -le 1 ]; then
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train_perturbed_mfcc data/lang exp/tri3b exp/tri3b_ali_perturbed_mfcc
fi


if [ $stage -le 2 ]; then
  steps/nnet2/train_block.sh --stage "$train_stage" \
     --num-jobs-nnet 4 --num-threads 1 --parallel-opts "$parallel_opts" \
     --bias-stddev 0.5 --splice-width 7 --egs-opts "--feat-type raw" \
     --softmax-learning-rate-factor 0.5 \
     --initial-learning-rate 0.04 --final-learning-rate 0.004 \
     --num-epochs-extra 10 --add-layers-period 3 --mix-up 4000 \
     --cmd "$decode_cmd" --hidden-layer-dim 450 \
      data/train_perturbed_fbank data/lang exp/tri3b_ali_perturbed_mfcc exp/nnet4b_gpu  || exit 1
fi


if [ $stage -le 3 ]; then
  # Create the testing data.
  featdir=`pwd`/mfcc
  mkdir -p $featdir
  fbank_conf=conf/fbank_40.conf
  echo "--num-mel-bins=40" > $fbank_conf
  for x in test_mar87 test_oct87 test_feb89 test_oct89 test_feb91 test_sep92 train; do
    mkdir -p data/${x}_fbank
    cp data/$x/* data/${x}_fbank || true
    steps/make_fbank.sh --fbank-config "$fbank_conf" --nj 8 \
      --cmd "run.pl" data/${x}_fbank exp/make_fbank/$x $featdir  || exit 1;
    steps/compute_cmvn_stats.sh data/${x}_fbank exp/make_fbank/$x $featdir  || exit 1;
  done
  utils/combine_data.sh data/test_fbank data/test_{mar87,oct87,feb89,oct89,feb91,sep92}_fbank
  steps/compute_cmvn_stats.sh data/test_fbank exp/make_fbank/test $featdir  
fi

if [ $stage -le 4 ]; then
   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --feat-type raw \
     exp/tri3b/graph data/test_fbank exp/nnet4b_gpu/decode
   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --feat-type raw \
     exp/tri3b/graph_ug data/test_fbank exp/nnet4b_gpu/decode_ug
fi
