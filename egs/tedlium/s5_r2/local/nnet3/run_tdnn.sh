#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.
# results below are for relu dim 768 - 3 epochs - 7 layers same splicing as chain
%WER 13.1 | 507 17792 | 88.8 7.8 3.4 1.9 13.1 84.6 | -0.123 | exp/nnet3/tdnn/decode_dev/score_11_0.5/ctm.filt.filt.sys
%WER 11.7 | 507 17792 | 90.3 7.0 2.7 2.0 11.7 81.9 | -0.270 | exp/nnet3/tdnn/decode_dev_rescore/score_10_0.0/ctm.filt.filt.sys
%WER 11.8 | 1155 27512 | 89.8 7.5 2.6 1.7 11.8 79.0 | -0.201 | exp/nnet3/tdnn/decode_test/score_10_0.0/ctm.filt.filt.sys
%WER 10.7 | 1155 27512 | 90.8 6.6 2.6 1.5 10.7 75.7 | -0.278 | exp/nnet3/tdnn/decode_test_rescore/score_10_0.0/ctm.filt.filt.sys


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=1
affix=
train_stage=-10
common_egs_dir=
reporting_email=
remove_egs=true
decode_iter=

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

dir=exp/nnet3/tdnn
dir=$dir${affix:+_$affix}
train_set=train_sp #_sp stands for speed-perturbed. This is hard-coded to speed 
                   # pertub data.
ali_dir=exp/tri3_ali_sp

local/nnet3/run_ivector_common.sh --stage $stage --generate-alignments true || exit 1;

#    --splice-indexes "-1,0,1 -1,0,1,2 -3,0,3 -3,0,3 -3,0,3 -6,-3,0" \
if [ $stage -le 9 ]; then
  echo "$0: creating neural net configs";

  # create the config files for nnet initialization
  python steps/nnet3/tdnn/make_configs.py  \
    --feat-dir data/${train_set}_hires \
    --ivector-dir exp/nnet3/ivectors_${train_set} \
    --ali-dir $ali_dir \
    --relu-dim 500 \
    --splice-indexes "-1,0,1 -1,0,1,2 -3,0,3 -3,0,3 -3,0,3 -6,-3,0 0" \
    --use-presoftmax-prior-scale true \
   $dir/configs || exit 1;
fi

if [ $stage -le 10 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{09,10,11,12}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 2 \
    --trainer.optimization.initial-effective-lrate 0.0015 \
    --trainer.optimization.final-effective-lrate 0.00015 \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 20 \
    --feat-dir=data/${train_set}_hires \
    --ali-dir $ali_dir \
    --lang data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;

fi

graph_dir=exp/tri3/graph
if [ $stage -le 11 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi

  for decode_set in dev test; do
    (
    steps/nnet3/decode.sh \
      --nj 4 --cmd "$decode_cmd" $iter_opts \
      --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
      $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter} || exit 1;

    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_test data/lang_rescore data/${decode_set}_hires \
      $dir/decode_${decode_set}${decode_iter:+_$decode_iter} \
      $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_rescore || exit 1;
    ) &
  done
fi
wait;
