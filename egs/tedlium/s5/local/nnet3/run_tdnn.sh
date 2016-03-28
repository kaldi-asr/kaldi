#!/bin/bash

set -uo pipefail
set -e

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice. 
# This script is designed to run using the 'cleaned-up' data.

# Run this script with the options --cleanup-affix <cleanup_affix> --min-seg-len <min_seg_len>, 
# where <cleanup_affix> matches the cleanup-affix passed to local/run_cleanup_segmentation.sh
# <min_seg_len> is required to combine segments so as to not lose any data when doing chain training or sequence training
# To run without any cleanup, pass the options --cleanup-affix "" --min-seg-len ""

# %WER 14.7 | 507 17792 | 88.0 8.8 3.2 2.7 14.7 89.7 | -0.154 | exp/nnet3/tdnn/decode_dev/score_9_0.5/ctm.filt.filt.sys
# %WER 13.4 | 507 17792 | 89.3 7.7 3.0 2.7 13.4 85.0 | -0.189 | exp/nnet3/tdnn/decode_dev_rescore/score_10_0.0/ctm.filt.filt.sys

# %WER 13.5 | 1155 27512 | 88.4 8.4 3.2 1.9 13.5 83.6 | -0.067 | exp/nnet3/tdnn/decode_test/score_11_0.5/ctm.filt.filt.sys
# %WER 12.3 | 1155 27512 | 89.8 7.6 2.6 2.2 12.3 79.4 | -0.152 | exp/nnet3/tdnn/decode_test_rescore/score_10_0.0/ctm.filt.filt.sys

## Cleanup results
## local/run_cleanup_segmentation.sh --cleanup-affix cleaned_b --pad-length 5 --max-silence-length 50 --max-incorrect-words 0 --min-correct-frames 0 --max-segment-wer 20000 --min-wer-for-splitting 10 --ngram-order 2 --interpolation-weights 0:0.1:0.9 --top-words-interpolation-weight 0.1 --hypothesis-method Oracle

# %WER 14.5 | 507 17792 | 87.8 8.6 3.6 2.3 14.5 88.6 | -0.083 | exp/nnet3_cleaned_b/tdnn/decode_dev/score_11_0.5/ctm.filt.filt.sys
# %WER 13.3 | 507 17792 | 89.3 7.8 3.0 2.5 13.3 86.2 | -0.215 | exp/nnet3_cleaned_b/tdnn/decode_dev_rescore/score_10_0.0/ctm.filt.filt.sys

# %WER 13.5 | 1155 27512 | 88.6 8.5 2.9 2.1 13.5 83.8 | -0.085 | exp/nnet3_cleaned_b/tdnn/decode_test/score_11_0.0/ctm.filt.filt.sys
# %WER 12.1 | 1155 27512 | 89.8 7.5 2.7 1.9 12.1 79.4 | -0.201 | exp/nnet3_cleaned_b/tdnn/decode_test_rescore/score_10_0.0/ctm.filt.filt.sys


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

min_seg_len=1.55
cleanup_affix=cleaned

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

dir=exp/nnet3${cleanup_affix:+_$cleanup_affix}/tdnn
dir=$dir${affix:+_$affix}

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.

train_set=train_${cleanup_affix}_sp_min${min_seg_len}

gmm_dir=exp/tri3_${cleanup_affix}
ali_dir=${gmm_dir}_ali_${train_set}   

local/nnet3/run_ivector_common.sh --stage $stage \
  --generate-alignments true \
  --min-seg-len $min_seg_len \
  --affix ${cleanup_affix:+_$cleanup_affix}

if [ $stage -le 11 ]; then
  echo "$0: creating neural net configs";

  # create the config files for nnet initialization
  repair_opts=${self_repair_scale:+" --self-repair-scale=$self_repair_scale "}
  
  python steps/nnet3/tdnn/make_configs.py  \
    $repair_opts \
    --feat-dir=data/${train_set}_hires \
    --ivector-dir=exp/nnet3${cleanup_affix:+_$cleanup_affix}/ivectors_${train_set} \
    --ali-dir=$ali_dir \
    --relu-dim=500 \
    --splice-indexes="-1,0,1 -1,0,1,2 -3,0,3 -3,0,3 -3,0,3 -6,-3,0" \
    --use-presoftmax-prior-scale=true \
   $dir/configs || exit 1;
fi

if [ $stage -le 12 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{09,10,11,12}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=exp/nnet3${cleanup_affix:+_$cleanup_affix}/ivectors_${train_set} \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs=4 \
    --trainer.optimization.num-jobs-initial=3 \
    --trainer.optimization.num-jobs-final=8 \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --egs.dir="$common_egs_dir" \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=20 \
    --feat-dir=data/${train_set}_hires \
    --ali-dir=$ali_dir \
    --lang=data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;

fi

graph_dir=$gmm_dir/graph
if [ $stage -le 13 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi

  for decode_set in dev test; do
    (
    steps/nnet3/decode.sh \
      --nj $(cat data/$decode_set/spk2utt | wc -l) --cmd "$decode_cmd" $iter_opts \
      --online-ivector-dir exp/nnet3${cleanup_affix:+_$cleanup_affix}/ivectors_${decode_set} \
      --scoring-opts "--min_lmwt 5 --max_lmwt 15" \
      $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter} || exit 1;

    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_test data/lang_rescore data/${decode_set}_hires \
      $dir/decode_${decode_set}${decode_iter:+_$decode_iter} \
      $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_rescore || exit 1;
    ) &
  done
fi

wait
