#!/bin/bash

# This is pnorm neural net training on top of adapted 40-dimensional features.


train_stage=-100
temp_dir=  # e.g. --temp-dir /export/m1-02/dpovey/kaldi-dan2/egs/wsj/s5/
dir=exp/nnet5d

# we derived this from run_5d.  Since cpu is slower than gpu, we increased the
# num-jobs from 4 to 8.  We doubled the final learning rate because we doubled
# the num-jobs, but we didn't double the initial learning rate as we were
# concerned it might become unstable.  [this is a bit of a black art].

. ./cmd.sh
. utils/parse_options.sh

( 

  if [ ! -z "$temp_dir" ] && [ ! -e $dir/egs ]; then
    mkdir -p $dir
    mkdir -p $temp_dir/$dir/egs
    ln -s $temp_dir/$dir/egs $dir/
  fi

  steps/nnet2/train_pnorm.sh --stage $train_stage \
   --num-jobs-nnet 8 \
   --mix-up 8000 \
   --initial-learning-rate 0.02 --final-learning-rate 0.004 \
   --num-hidden-layers 4 \
   --pnorm-input-dim 2000 --pnorm-output-dim 400 \
   --cmd "$decode_cmd" \
   --p 2 \
    data/train_si284 data/lang exp/tri4b_ali_si284 $dir || exit 1

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 10 \
    --transform-dir exp/tri4b/decode_tgpr_dev93 \
     exp/tri4b/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 8 \
    --transform-dir exp/tri4b/decode_tgpr_eval92 \
     exp/tri4b/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 10 \
    --transform-dir exp/tri4b/decode_bd_tgpr_dev93 \
     exp/tri4b/graph_bd_tgpr data/test_dev93 $dir/decode_bd_tgpr_dev93

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 8 \
    --transform-dir exp/tri4b/decode_bd_tgpr_eval92 \
     exp/tri4b/graph_bd_tgpr data/test_eval92 $dir/decode_bd_tgpr_eval92
)
