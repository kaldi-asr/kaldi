#!/bin/bash

# Note: In order to run BNF, run run_bnf.sh
. ./path.sh
. ./cmd.sh

set -e
set -o pipefail
set -u

. utils/parse_options.sh
use_gpu=true
if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
  fi
  parallel_opts="-l gpu=1" 
  num_threads=1
  minibatch_size=512
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  parallel_opts="-pe smp $num_threads" 
  minibatch_size=128
fi

bnf_train_stage=-100
align_dir=exp/tri4b_ali_si284 
if [ ! -f exp_bnf/tri6_bnf/.done ]; then
  mkdir -p exp_bnf
  mkdir -p exp_bnf/tri6_bnf
  echo ---------------------------------------------------------------------
  echo "Starting training the bottleneck network"
  echo ---------------------------------------------------------------------
  steps/nnet2/train_tanh_bottleneck.sh \
    --stage $bnf_train_stage --num-jobs-nnet 4 \
    --num-threads $num_threads --mix-up 5000 --max-change 40 \
    --minibatch-size $minibatch_size --parallel-opts "$parallel_opts" \
    --cmd "$train_cmd -l mem_free=2G,ram_free=2G"\
    --initial-learning-rate 0.005 \
    --final-learning-rate 0.0005 \
    --num-hidden-layers 5 \
    --bottleneck-dim 42 --hidden-layer-dim 1024 \
    data/train_si284 data/lang $align_dir exp_bnf/tri6_bnf || exit 1 
  touch exp_bnf/tri6_bnf/.done
fi

[ ! -d param_bnf ] && mkdir -p param_bnf
if [ ! -f data_bnf/train_bnf/.done ]; then
  mkdir -p data_bnf
  # put the archives in param_bnf/.
  steps/nnet2/dump_bottleneck_features.sh --cmd "$train_cmd" \
    --transform-dir exp/tri4a  data/train_si284 data_bnf/train_bnf exp_bnf/tri6_bnf param_bnf exp_bnf/dump_bnf
  touch data_bnf/train_bnf/.done
fi 

[ ! -d data/test_eval92 ] && echo "No such directory data/test_eval92" && exit 1;
[ ! -d data/test_dev93 ] && echo "No such directory data/test_dev93" && exit 1;
[ ! -d exp/tri4b/decode_bd_tgpr_eval92 ] && echo "No such directory exp/tri4b/decode_bd_tgpr_eval92" && exit 1;
[ ! -d exp/tri4b/decode_bd_tgpr_dev93 ] && echo "No such directory exp/tri4b/decode_bd_tgpr_dev93" && exit 1;
# put the archives in param_bnf/.
steps/nnet2/dump_bottleneck_features.sh --nj 8 \
  --transform-dir exp/tri4b/decode_bd_tgpr_eval92 data/test_eval92 data_bnf/eval92_bnf exp_bnf/tri6_bnf param_bnf exp_bnf/dump_bnf

steps/nnet2/dump_bottleneck_features.sh --nj 10 \
  --transform-dir exp/tri4b/decode_bd_tgpr_dev93 data/test_dev93 data_bnf/dev93_bnf exp_bnf/tri6_bnf param_bnf exp_bnf/dump_bnf



if [ ! data_bnf/train/.done -nt data_bnf/train_bnf/.done ]; then
  steps/nnet/make_fmllr_feats.sh --cmd "$train_cmd --max-jobs-run 10" \
     --transform-dir $align_dir  data_bnf/train_sat data/train_si284 \
    exp/tri4b exp_bnf/make_fmllr_feats/log param_bnf/ 

  steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
    data_bnf/train_bnf data_bnf/train_sat data_bnf/train \
    exp_bnf/append_feats/log param_bnf/ 
  steps/compute_cmvn_stats.sh --fake data_bnf/train exp_bnf/make_fmllr_feats param_bnf
  rm -r data_bnf/train_sat

  touch data_bnf/train/.done
fi
## preparing Bottleneck features for eval92 and dev93
steps/nnet/make_fmllr_feats.sh \
  --nj 8 --transform-dir exp/tri4b/decode_bd_tgpr_eval92 data_bnf/eval92_sat data/test_eval92 \
  $align_dir exp_bnf/make_fmllr_feats/log param_bnf/ 
steps/nnet/make_fmllr_feats.sh \
  --nj 10 --transform-dir exp/tri4b/decode_bd_tgpr_dev93 data_bnf/dev93_sat data/test_dev93 \
  $align_dir exp_bnf/make_fmllr_feats/log param_bnf/ 

steps/append_feats.sh --nj 4 \
  data_bnf/eval92_bnf data_bnf/eval92_sat data_bnf/eval92 \
  exp_bnf/append_feats/log param_bnf/ 
steps/append_feats.sh --nj 4 \
  data_bnf/dev93_bnf data_bnf/dev93_sat data_bnf/dev93 \
  exp_bnf/append_feats/log param_bnf/ 
  
steps/compute_cmvn_stats.sh --fake data_bnf/eval92 exp_bnf/make_fmllr_feats param_bnf
steps/compute_cmvn_stats.sh --fake data_bnf/dev93 exp_bnf/make_fmllr_feats param_bnf

rm -r data_bnf/eval92_sat
rm -r data_bnf/dev93_sat

# you can run sgmm2 on this bottleneck features
local/run_bnf_sgmm.sh

exit 0;
