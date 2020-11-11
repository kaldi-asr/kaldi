#!/usr/bin/env bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

# Results (2 epochs):
# Number of parameters: 6056880
# %WER 15.3 | 507 17792 | 87.4 9.0 3.6 2.7 15.3 90.1 | -0.081 | exp/nnet3/tdnn_sp/decode_dev/score_10_0.5/ctm.filt.filt.sys
# %WER 13.9 | 507 17792 | 88.4 8.0 3.6 2.3 13.9 85.8 | -0.164 | exp/nnet3/tdnn_sp/decode_dev_rescore/score_10_0.5/ctm.filt.filt.sys
# %WER 13.8 | 1155 27512 | 88.5 8.7 2.7 2.3 13.8 84.2 | -0.076 | exp/nnet3/tdnn_sp/decode_test/score_10_0.0/ctm.filt.filt.sys
# %WER 12.5 | 1155 27512 | 89.6 7.7 2.6 2.1 12.5 81.5 | -0.133 | exp/nnet3/tdnn_sp/decode_test_rescore/score_10_0.0/ctm.filt.filt.sys

# 4 epochs
# %WER 14.6 | 507 17792 | 87.9 8.7 3.4 2.5 14.6 88.6 | -0.111 | exp/nnet3/tdnn/decode_dev/score_10_0.5/ctm.filt.filt.sys
# %WER 13.2 | 507 17792 | 89.4 7.7 2.9 2.6 13.2 85.0 | -0.170 | exp/nnet3/tdnn/decode_dev_rescore/score_10_0.0/ctm.filt.filt.sys
# %WER 13.5 | 1155 27512 | 88.7 8.5 2.7 2.3 13.5 83.6 | -0.110 | exp/nnet3/tdnn/decode_test/score_10_0.0/ctm.filt.filt.sys
# %WER 12.1 | 1155 27512 | 89.9 7.5 2.6 2.1 12.1 80.3 | -0.178 | exp/nnet3/tdnn/decode_test_rescore/score_10_0.0/ctm.filt.filt.sys

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

if [ $stage -le 9 ]; then
  echo "$0: creating neural net configs";

  # create the config files for nnet initialization
  python steps/nnet3/tdnn/make_configs.py  \
    --feat-dir data/${train_set}_hires \
    --ivector-dir exp/nnet3/ivectors_${train_set} \
    --ali-dir $ali_dir \
    --relu-dim 500 \
    --splice-indexes "-1,0,1 -1,0,1,2 -3,0,3 -3,0,3 -3,0,3 -6,-3,0" \
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
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 8 \
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
      --nj $(wc -l < data/$decode_set/spk2utt) --cmd "$decode_cmd" $iter_opts \
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
