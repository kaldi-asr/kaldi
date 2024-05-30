#!/usr/bin/env bash

# This is to be run after run_nnet2.sh

. ./cmd.sh

use_preconditioning=true

stage=1
train_stage=-10
use_gpu=true
. ./cmd.sh
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
  gpu_opts="--gpu 1"
  train_parallel_opts="--gpu 1"
  num_threads=1
  # the _a is in case I want to change the parameters.
  srcdir=exp/nnet2_online/nnet_a_gpu
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  gpu_opts=""
  num_threads=16
  train_parallel_opts="--num-threads 16"
  srcdir=exp/nnet2_online/nnet_a
fi


set -e

nj=40

if [ $stage -le 1 ]; then
  # the make_denlats job is always done on CPU not GPU, since in any case
  # the graph search and lattice determinization takes quite a bit of CPU.
  # note: it's the sub-split option that determinies how many jobs actually
  # run at one time.
  steps/nnet2/make_denlats.sh --cmd "$decode_cmd --mem 1G" \
      --nj $nj --sub-split 40 --num-threads 6 --parallel-opts "--num-threads 6" \
      --online-ivector-dir exp/nnet2_online/ivectors_train \
      data/train_hires data/lang $srcdir ${srcdir}_denlats
fi

if [ $stage -le 2 ]; then
  if $use_gpu; then use_gpu_opt=yes; else use_gpu_opt=no; fi
  steps/nnet2/align.sh  --cmd "$decode_cmd $gpu_opts" \
      --online-ivector-dir exp/nnet2_online/ivectors_train \
      --use-gpu $use_gpu_opt \
      --nj $nj data/train_hires data/lang ${srcdir} ${srcdir}_ali
fi

if [ $stage -le 3 ]; then
  if [ $USER == dpovey ]; then # this shows how you can split across multiple file-systems.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/dpovey/kaldi-online/egs/fisher_english/s5/${srcdir}_smbr/degs ${srcdir}_smbr/degs/storage
  fi
  # decreasing the learning rate by a factor of 2, due to having so much data,
  # and decreasing the number of epochs for the same reason.
  # the io-opts option is to have more get_egs (and similar) jobs running at a time,
  # since we're using 4 disks.
  steps/nnet2/train_discriminative.sh --cmd "$decode_cmd" --learning-rate 0.00001 \
    --io-opts "--num-threads 10" \
    --num-epochs 4 \
    --use-preconditioning $use_preconditioning \
    --online-ivector-dir exp/nnet2_online/ivectors_train \
    --num-jobs-nnet 4  --num-threads $num_threads --parallel-opts "$gpu_opts" \
      data/train_hires data/lang \
    ${srcdir}_ali ${srcdir}_denlats ${srcdir}/final.mdl ${srcdir}_smbr
fi

if [ $stage -le 4 ]; then
  # we'll do the decoding as 'online' decoding by using the existing
  # _online directory but with extra models copied to it.
  for epoch in 1 2 3 4; do
    cp ${srcdir}_smbr/epoch${epoch}.mdl ${srcdir}_online/smbr_epoch${epoch}.mdl
  done

  for epoch in 1 2 3 4; do
    # do the actual online decoding with iVectors, carrying info forward from
    # previous utterances of the same speaker.
    steps/online/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 --iter smbr_epoch${epoch} \
       exp/tri5a/graph data/dev ${srcdir}_online/decode_dev_smbr_epoch${epoch} || exit 1;
  done
fi

wait

# for results, see the end of run_nnet2.sh
