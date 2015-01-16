#!/bin/bash

# This is p-norm neural net training, with the "fast" script, on top of adapted
# 40-dimensional features.
# This version uses 460 hours of "clean" (typically relatively un-accented) 
# training data.
# We're using 6 jobs rather than 4, for speed.

# Note: we highly discourage running this with --use-gpu false, it will
# take way too long.

train_stage=-10
use_gpu=true

. cmd.sh
. ./path.sh
. utils/parse_options.sh


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
  dir=exp/nnet6a_clean_460_gpu
else
  # with just 4 jobs this might be a little slow.
  num_threads=16
  parallel_opts="-pe smp $num_threads" 
  minibatch_size=128
  dir=exp/nnet6a_clean_460
fi

. ./cmd.sh
. utils/parse_options.sh

if [ ! -f $dir/final.mdl ]; then
  if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then 
     # spread the egs over various machines.  will help reduce overload of any
     # one machine.
     utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/librispeech/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
   --samples-per-iter 400000 \
   --num-epochs 7 --num-epochs-extra 3 \
   --parallel-opts "$parallel_opts" \
   --num-threads "$num_threads" \
   --minibatch-size "$minibatch_size" \
   --num-jobs-nnet 6  --mix-up 10000 \
   --initial-learning-rate 0.01 --final-learning-rate 0.001 \
   --num-hidden-layers 4 \
   --pnorm-input-dim 4000 --pnorm-output-dim 400 \
   --cmd "$decode_cmd" \
    data/train_clean_460 data/lang exp/tri5b $dir || exit 1
fi


for test in test_clean test_other dev_clean dev_other; do
  steps/nnet2/decode.sh --nj 20 --cmd "$decode_cmd" \
    --transform-dir exp/tri5b/decode_pp_tgsmall_$test \
    exp/tri5b/graph_pp_tgsmall data/$test $dir/decode_pp_tgsmall_$test || exit 1;
  steps/lmrescore.sh --cmd "$decode_cmd" data/lang_pp_test_{tgsmall,tgmed} \
    data/$test $dir/decode_pp_{tgsmall,tgmed}_$test  || exit 1;
  steps/lmrescore_const_arpa.sh \
    --cmd "$decode_cmd" data/lang_pp_test_{tgsmall,tglarge} \
    data/$test $dir/decode_pp_{tgsmall,tglarge}_$test || exit 1;
  steps/lmrescore_const_arpa.sh \
    --cmd "$decode_cmd" data/lang_pp_test_{tgsmall,fglarge} \
    data/$test $dir/decode_pp_{tgsmall,fglarge}_$test || exit 1;
done

exit 0;
