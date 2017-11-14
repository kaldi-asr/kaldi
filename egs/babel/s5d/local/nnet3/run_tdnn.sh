#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. ./cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
train_stage=-10
speed_perturb=true
affix=
splice_indexes="-2,-1,0,1,2 -1,2 -3,3 -7,2 0"
common_egs_dir=
relu_dim=850
remove_egs=true

[ ! -f ./lang.conf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

[ -f local.conf ] && . ./local.conf

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

train_nj=72
export boost_sil use_pitch train_nj
export numLeavesMLLT numGaussMLLT


suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi
dir=exp/nnet3/tdnn
dir=$dir${affix:+_$affix}
dir=${dir}$suffix
train_set=train$suffix
ali_dir=exp/tri5_ali$suffix

local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb || exit 1;
if [ $stage -le 9 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/babel-$(date +'%m_%d_%H_%M')/s5d/$RANDOM/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_tdnn.sh --stage $train_stage \
    --num-epochs 5 --num-jobs-initial 2 --num-jobs-final 6 \
    --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate 0.0017 --final-effective-lrate 0.00017 \
    --cmd "$decode_cmd" \
    --egs-dir "$common_egs_dir" \
    --relu-dim $relu_dim \
    --remove-egs $remove_egs \
    data/${train_set}_hires data/langp/tri5_ali $ali_dir $dir  || exit 1;
  touch $dir/.done
fi


exit 0;

