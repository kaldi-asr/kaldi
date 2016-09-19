#!/bin/bash

# This is an ensemble training recipe using pnorm neural nets on top of adapted 40-dimensional features.
ensemble_size=4
initial_beta=0.1
final_beta=5


train_stage=-10
temp_dir=  # e.g. --temp-dir /export/m1-02/dpovey/kaldi-dan2/egs/wsj/s5/
parallel_opts="-l gpu=1,hostname=g*"  # This is suitable for the CLSP network, you'll likely have to change it.
dir=exp/nnet5e_gpu

# Note: since we multiplied the num-jobs by 1/4, we halved the
# learning rate, relative to run_5c.sh
. ././cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
. utils/parse_options.sh

(

  if [ ! -z "$temp_dir" ] && [ ! -e $dir/egs ]; then
    mkdir -p $dir
    mkdir -p $temp_dir/$dir/egs
    ln -s $temp_dir/$dir/egs $dir/
  fi

  steps/nnet2/train_pnorm_ensemble.sh --stage $train_stage \
   --num-jobs-nnet 4 --num-threads 1 --parallel-opts "$parallel_opts" \
   --mix-up 8000 \
   --initial-learning-rate 0.02 --final-learning-rate 0.002 \
   --num-hidden-layers 4 \
   --pnorm-input-dim 2000 --pnorm-output-dim 400 \
   --cmd "$decode_cmd" \
   --p 2 \
   --ensemble-size $ensemble_size --initial-beta $initial_beta --final-beta $final_beta \
    data/train_si284 data/lang exp/tri4b_ali_si284 $dir || exit 1

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 \
    --transform-dir exp/tri4b/decode_tgpr_dev93 \
     exp/tri4b/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 8 \
    --transform-dir exp/tri4b/decode_tgpr_eval92 \
     exp/tri4b/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 \
    --transform-dir exp/tri4b/decode_bd_tgpr_dev93 \
     exp/tri4b/graph_bd_tgpr data/test_dev93 $dir/decode_bd_tgpr_dev93

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 8 \
    --transform-dir exp/tri4b/decode_bd_tgpr_eval92 \
     exp/tri4b/graph_bd_tgpr data/test_eval92 $dir/decode_bd_tgpr_eval92
)
