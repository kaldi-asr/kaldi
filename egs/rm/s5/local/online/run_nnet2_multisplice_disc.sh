#!/usr/bin/env bash

# This is to be run after run_nnet2_multisplice.sh.
# It demonstrates discriminative training for the online-nnet2 models

. ./cmd.sh


stage=1
train_stage=-10
use_gpu=true
srcdir=exp/nnet2_online/nnet_ms_a_online
criterion=smbr
learning_rate=0.0016

drop_frames=false # only relevant for MMI

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ ! -f $srcdir/final.mdl ]; then
  echo "$0: expected $srcdir/final.mdl to exist; first run run_nnet2_multisplice.sh."
  exit 1;
fi

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  parallel_opts="--num-threads $num_threads"
fi

if [ $stage -le 1 ]; then
  # the conf/decode.config gives it higher than normal beam/lattice-beam of (20,10), since
  # otherwise on RM we'd get very thin lattices.
  nj=30
  num_threads_denlats=6
  steps/online/nnet2/make_denlats.sh --cmd "$decode_cmd --mem 1G --num-threads $num_threads_denlats" \
      --nj $nj --sub-split 40 --num-threads "$num_threads_denlats" --config conf/decode.config \
     data/train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 2 ]; then
  # hardcode no-GPU for alignment, although you could use GPU [you wouldn't
  # get excellent GPU utilization though.]
  nj=100
  use_gpu=no
  gpu_opts=
  steps/online/nnet2/align.sh  --cmd "$decode_cmd $gpu_opts" --use-gpu "$use_gpu" \
      --nj $nj data/train data/lang $srcdir ${srcdir}_ali || exit 1;
fi


if [ $stage -le 3 ]; then
  # I tested the following with  --max-temp-archives 3
  # to test other branches of the code.
  # the --max-jobs-run 5 limits the I/O.
  steps/online/nnet2/get_egs_discriminative2.sh \
    --cmd "$decode_cmd --max-jobs-run 5" \
    --criterion $criterion --drop-frames $drop_frames \
     data/train data/lang ${srcdir}{_ali,_denlats,,_degs} || exit 1;
fi

if [ $stage -le 4 ]; then
  steps/nnet2/train_discriminative2.sh --cmd "$decode_cmd $parallel_opts" \
    --learning-rate $learning_rate \
    --criterion $criterion --drop-frames $drop_frames \
    --num-epochs 6 \
    --num-jobs-nnet 2 --num-threads $num_threads \
      ${srcdir}_degs ${srcdir}_${criterion}_${learning_rate} || exit 1;
fi

if [ $stage -le 5 ]; then
  ln -sf $(utils/make_absolute.sh $srcdir/conf) ${srcdir}_${criterion}_${learning_rate}/conf # so it acts like an online-decoding directory

  for epoch in 0 1 2 3 4 5 6; do
    steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
      --iter epoch$epoch exp/tri3b/graph data/test ${srcdir}_${criterion}_${learning_rate}/decode_epoch$epoch &
    steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
      --iter epoch$epoch exp/tri3b/graph_ug data/test ${srcdir}_${criterion}_${learning_rate}/decode_ug_epoch$epoch &
  done
  wait
  for dir in ${srcdir}_${criterion}_${learning_rate}/decode*; do grep WER $dir/wer_* | utils/best_wer.sh; done
fi
