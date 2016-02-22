#!/bin/bash


# This is discriminative training, to be run after run_nnet2.sh.

. cmd.sh


stage=1
train_stage=-10
use_gpu=true
srcdir=exp/nnet2_online/nnet_ms_a

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
  gpu_opts="-l gpu=1"
  train_parallel_opts="-l gpu=1"
  num_threads=1
  # the _a is in case I want to change the parameters.
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  gpu_opts=""
  num_threads=16
  train_parallel_opts="-pe smp 16"
fi

nj=40

if [ $stage -le 1 ]; then
 
  # the make_denlats job is always done on CPU not GPU, since in any case
  # the graph search and lattice determinization takes quite a bit of CPU.
  # note: it's the sub-split option that determinies how many jobs actually
  # run at one time.
  steps/nnet2/make_denlats.sh --cmd "$decode_cmd -l mem_free=1G,ram_free=1G" \
      --nj $nj --sub-split 40 --num-threads 6 --parallel-opts "-pe smp 6" \
      --online-ivector-dir exp/nnet2_online/ivectors_train_si284 \
      data/train_si284_hires data/lang $srcdir ${srcdir}_denlats
fi

if [ $stage -le 2 ]; then
  if $use_gpu; then gpu_opt=yes; else gpu_opt=no; fi
  steps/nnet2/align.sh  --cmd "$decode_cmd $gpu_opts" \
      --online-ivector-dir exp/nnet2_online/ivectors_train_si284 \
      --use-gpu $gpu_opt \
      --nj $nj data/train_si284_hires data/lang ${srcdir} ${srcdir}_ali  || exit 1;
fi

if [ $stage -le 3 ]; then
  steps/nnet2/train_discriminative.sh --cmd "$decode_cmd" --learning-rate 0.0002 \
    --stage $train_stage \
    --online-ivector-dir exp/nnet2_online/ivectors_train_si284 \
    --num-jobs-nnet 4  --num-threads $num_threads --parallel-opts "$gpu_opts" \
    data/train_si284_hires data/lang \
    ${srcdir}_ali ${srcdir}_denlats ${srcdir}/final.mdl ${srcdir}_smbr || exit 1;
fi

if [ $stage -le 4 ]; then
  # we'll do the decoding as 'online' decoding by using the existing
  # _online directory but with extra models copied to it.
  for epoch in 1 2 3 4; do
    cp ${srcdir}_smbr/epoch${epoch}.mdl ${srcdir}_online/smbr_epoch${epoch}.mdl
  done

  error_file=$(dirname $srcdir)/.error.decode_smbr
  rm $error_file 2>/dev/null || true

  for epoch in 1 2 3 4; do
    # do the actual online decoding with iVectors, carrying info forward from 
    # previous utterances of the same speaker.
    # We just do the bd_tgpr decodes; otherwise the number of combinations 
    # starts to get very large.
    for lm_suffix in bd_tgpr; do
      graph_dir=exp/tri4b/graph_${lm_suffix}
      for year in eval92 dev93; do
        steps/online/nnet2/decode.sh --cmd "$decode_cmd" --nj 8 --iter smbr_epoch${epoch} \
          "$graph_dir" data/test_${year} ${srcdir}_online/decode_${lm_suffix}_${year}_smbr_epoch${epoch} || touch $error_file &
      done
    done
  done
  wait
  [ -f $error_file ] && echo "$0: error decoding the SMBR systems." && exit 1;
fi

