#!/bin/bash

###########################################################################################
# This script was copied from egs/fisher_swbd/s5/local/nnet3/run_tdnn.sh
# The source commit was e69198c3dc5633f98eb88e1cdf20b2521a598f21
# Changes made:
#  - Modified paths to match multi_en naming conventions
###########################################################################################

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. ./cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.

stage=0
affix=
train_stage=-10
common_egs_dir=
reporting_email=
remove_egs=true
multi=multi_a

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

dir=exp/$multi/nnet3/tdnn
dir=$dir${affix:+_$affix}
train_set=$multi/tdnn
ali_dir=exp/$multi/tri5_ali

local/nnet3/run_ivector_common.sh --stage $stage --multi $multi \
  --speed-perturb false || exit 1;

if [ $stage -le 9 ]; then
  echo "$0: creating neural net configs";

  # create the config files for nnet initialization
  python steps/nnet3/tdnn/make_configs.py  \
    --feat-dir data/${train_set}_hires \
    --ivector-dir exp/$multi/nnet3/ivectors_${train_set} \
    --ali-dir $ali_dir \
    --relu-dim 1024 \
    --splice-indexes "-2,-1,0,1,2 -1,2 -3,3 -3,3 -7,2 0"  \
    --use-presoftmax-prior-scale true \
   $dir/configs || exit 1;
fi


if [ $stage -le 10 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/multi_en-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir exp/$multi/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.0017 \
    --trainer.optimization.final-effective-lrate 0.00017 \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 500 \
    --use-gpu true \
    --feat-dir=data/${train_set}_hires \
    --ali-dir $ali_dir \
    --lang data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;

fi

graph_dir=exp/$multi/tri5/graph_tg
if [ $stage -le 11 ]; then
  for decode_set in eval2000 rt03; do
    (
    num_jobs=`cat data/${decode_set}_hires/test/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
        --online-ivector-dir exp/$multi/nnet3/ivectors_${decode_set} \
       $graph_dir data/${decode_set}_hires/test $dir/decode_${decode_set}_tg || exit 1;
#    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
#        data/lang_fsh_sw1_{tg,fg} data/${decode_set}_hires \
#       $dir/decode_${decode_set}_fsh_sw1_{tg,fg} || exit 1;
    ) &
  done
fi
wait;
exit 0;

