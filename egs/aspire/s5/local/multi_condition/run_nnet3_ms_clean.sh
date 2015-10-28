#!/bin/bash

. cmd.sh


stage=7
train_stage=-10
use_gpu=true
dir=exp/nnet3_multicondition/nnet_ms_clean

set -e
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="-l gpu=1"
  num_threads=1
  minibatch_size=512

  if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
    parallel_opts="$parallel_opts --config conf/queue_no_k20.conf --allow-k20 false"
    # that config is like the default config in the text of queue.pl, but adding the following lines.
    # default allow_k20=true
    # option allow_k20=true
    # option allow_k20=false -l 'hostname=!g01&!g02&!b06'
    # It's a workaround for an NVidia CUDA library bug for our currently installed version
    # of the CUDA toolkit, that only shows up on k20's
  fi
  # the _a is in case I want to change the parameters.
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
fi

# do the common parts of the script.
local/multi_condition/run_nnet2_common_clean.sh --stage $stage


if [ $stage -le 7 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/fisher_english_reverb-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_tdnn.sh --stage $train_stage \
    --num-epochs 1 --num-jobs-initial 4 --num-jobs-final 22 \
    --splice-indexes "-2,-1,0,1,2 -1,2 -3,3 -7,2 0" \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_multicondition/ivectors_train_clean \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --minibatch-size "$minibatch_size" \
    --initial-effective-lrate 0.0015 --final-effective-lrate 0.00015 \
    --cmd "$decode_cmd" \
    --relu-dim 1024 \
    --frames-per-eg 16 \
    --remove-egs false \
    data/train_hires data/lang exp/tri5a $dir  || exit 1;
fi

if [ $stage -le 8 ]; then
  # dump iVectors for the testing data.
  for data_dir in dev test; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
      data/${data_dir}_hires exp/nnet2_multicondition/extractor_clean exp/nnet2_multicondition/ivectors_${data_dir}_clean || exit 1;
  done
fi


if [ $stage -le 9 ]; then
  # this does offline decoding that should give about the same results as the
  # real online decoding (the one with --per-utt true)
  for data_dir in dev test; do
   ( steps/nnet3/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
      --online-ivector-dir exp/nnet2_multicondition/ivectors_${data_dir}_clean \
      exp/tri5a/graph data/${data_dir}_hires $dir/decode_${data_dir} || exit 1;
   ) &
  done
  wait;
fi

exit 0;
