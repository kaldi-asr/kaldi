#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.
set -e

stage=0
train_stage=-10
use_sat_alignments=true
affix=
speed_perturb=true
common_egs_dir=

# training options
initial_effective_lrate=0.0015
final_effective_lrate=0.00015
num_epochs=4
num_jobs_initial=2
num_jobs_final=12
remove_egs=true

# feature options
use_ivectors=true

# End configuration section.

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

dir=exp/nnet3/tdnn${speed_perturb:+_sp}${affix:+_$affix}
if [ "$use_sat_alignments" == "true" ] ; then
  gmm_dir=exp/tri5a
else
  gmm_dir=exp/tri3a
fi

if [ "$speed_perturb" == "true" ]; then
  train_set=train_sp
  ali_dir=${gmm_dir}_sp_ali
else
  train_set=train
  ali_dir=${gmm_dir}_ali
fi

graph_dir=$gmm_dir/graph

if [ $stage -le 7 ]; then
  local/nnet3/run_ivector_common.sh --stage $stage \
    --use-sat-alignments $use_sat_alignments \
    --speed-perturb $speed_perturb || exit 1;
fi

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/hkust-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  if [ "$use_ivectors" == "true" ]; then
    ivector_opts=" --online-ivector-dir exp/nnet3/ivectors_${train_set}_hires "
    cmvn_opts="--norm-means=false --norm-vars=false"
  else
    ivector_opts=
    cmvn_opts="--norm-means=true --norm-vars=true"
  fi

  steps/nnet3/train_tdnn.sh $ivector_opts --stage $train_stage \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --splice-indexes "-2,-1,0,1,2 -1,2 -3,3 -7,2 -3,3 0 0" \
    --feat-type raw --get-egs-stage 5 \
    --cmvn-opts "$cmvn_opts" \
    --io-opts "--max-jobs-run 12" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --cmd "$decode_cmd" \
    --relu-dim 850 \
    --egs-dir "$common_egs_dir" \
    --remove-egs $remove_egs \
    data/${train_set}_hires data/lang $ali_dir $dir  || exit 1;
fi

if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for decode_set in dev; do
      (
      num_jobs=`cat data/${decode_set}/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      decode_dir=${dir}/decode_${decode_set}
      if [ "$use_ivectors" == "true" ]; then
        ivector_opts=" --online-ivector-dir exp/nnet3/ivectors_${decode_set} "
      else
        ivector_opts=
      fi

      steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" $ivector_opts \
         $graph_dir data/${decode_set}_hires $decode_dir || exit 1;
      ) &
  done
fi
wait;
exit 0;
