#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=9
dir=
has_fisher=true
mic=ihm
use_sat_alignments=true
affix=
speed_perturb=true
common_egs_dir=

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

multidir=$1
virtualdir=$2
num_outputs=$3
train_stage=$4

echo dir is $dir

if ! cuda-compiled; then
  cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{1,2,5,6}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

#  train_stage=`ls $dir | grep .mdl | sed s=.mdl==g | sort -n | tail -n 1`
  echo train_stage $train_stage

  train_set=train
  steps/nnet3/train_tdnn_joint.sh --stage $train_stage \
    --cleanup false \
    --num-outputs $num_outputs \
    --num-epochs 3 --num-jobs-initial 2 --num-jobs-final 12 \
    --splice-indexes "-2,-1,0,1,2 -1,2 -3,3 -7,2 -3,3 0 0" \
    --feat-type raw \
    --online-ivector-dir exp/$mic/nnet3/ivectors_${train_set}_hires \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --egs-dir "$common_egs_dir" \
    --initial-effective-lrate 0.0015 --final-effective-lrate 0.00015 \
    --cmd "$decode_cmd" \
    --relu-dim 850 \
    --tree-mapping $virtualdir/tree-mapping \
    data/$mic/${train_set}_hires data/lang $multidir/tree $virtualdir/ $dir  || exit 1;
fi

if [ $stage -le 9 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  for decode_set in dev eval; do
    num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    decode_dir=${dir}/decode_${decode_set}
    graph_dir=${virtualdir}/graph_ami_fsh.o3g.kn.pr1-7
    # use already-built graphs.
(      steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
          --online-ivector-dir exp/$mic/nnet3/ivectors_${decode_set} \
         $graph_dir data/$mic/${decode_set}_hires \
         $decode_dir || exit 1; ) &
    wait
    echo done
  done
fi

