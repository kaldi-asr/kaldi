#!/bin/bash

# This script is based on run_nnet2_multisplice.sh in
# egs/fisher_english/s5/local/online. It has been modified
# for language recognition.

. cmd.sh


stage=1
train_stage=-10
use_gpu=true
set -e
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


# assume use_gpu=true since it would be way too slow otherwise.

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
dir=exp/nnet2_online/nnet_ms_a
mkdir -p exp/nnet2_online


# Stages 1 through 5 are done in run_nnet2_common.sh,
# so it can be shared with other similar scripts.
local/dnn/run_nnet2_common.sh --stage $stage

if [ $stage -le 6 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
    utils/create_split_dir.pl /export/b0{6,7,8,9}/$(USER)/kaldi-data/egs/lre07/v2/$dir/egs/storage
  fi
  
  # Because we have a lot of data here and we don't want the training to take
  # too long, we reduce the number of epochs from the defaults (15 + 5) to (3 +
  # 1).  The option "--io-opts '-tc 12'" is to have more than the default number
  # (5) of jobs dumping the egs to disk; this is OK since we're splitting our
  # data across four filesystems for speed.


  lid/nnet2/train_multisplice_accel2.sh --stage $train_stage \
    --feat-type raw \
    --splice-indexes "layer0/-2:-1:0:1:2 layer1/-1:2 layer3/-3:3 layer4/-7:2" \
    --num-epochs 6 \
    --num-hidden-layers 6 \
    --num-jobs-initial 3 --num-jobs-final 18 \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --mix-up 10500 \
    --initial-effective-lrate 0.0015 --final-effective-lrate 0.00015 \
    --cmd "$decode_cmd" \
    --egs-dir "$common_egs_dir" \
    --pnorm-input-dim 3500 \
    --pnorm-output-dim 350 \
    data/train_hires_asr data/lang exp/tri5a $dir  || exit 1;

fi

exit 0;

