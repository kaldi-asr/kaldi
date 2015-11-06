#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.
set -e

stage=0
train_stage=-10
has_fisher=true
mic=ihm
use_sat_alignments=true
affix=
speed_perturb=true
common_egs_dir=

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

dir=exp/$mic/nnet3/tdnn${speed_perturb:+_sp}${affix:+_$affix}
if [ "$use_sat_alignments" == "true" ] ; then
  gmm_dir=exp/$mic/tri4a
else
  gmm_dir=exp/$mic/tri3a
fi

if [ "$speed_perturb" == "true" ]; then
  train_set=train_sp
  ali_dir=${gmm_dir}_sp_ali
else
  train_set=train
  ali_dir=${gmm_dir}_ali
fi

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$gmm_dir/graph_${LM}

local/nnet3/run_ivector_common.sh --stage $stage \
  --mic $mic \
  --use-sat-alignments $use_sat_alignments \
  --speed-perturb $speed_perturb || exit 1;

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_tdnn.sh --stage $train_stage \
    --num-epochs 3 --num-jobs-initial 2 --num-jobs-final 12 \
    --splice-indexes "-2,-1,0,1,2 -1,2 -3,3 -7,2 -3,3 0 0" \
    --feat-type raw \
    --online-ivector-dir exp/$mic/nnet3/ivectors_${train_set}_hires \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --io-opts "--max-jobs-run 12" \
    --egs-dir "$common_egs_dir" \
    --initial-effective-lrate 0.0015 --final-effective-lrate 0.00015 \
    --cmd "$decode_cmd" \
    --relu-dim 850 \
    data/$mic/${train_set}_hires data/lang $ali_dir $dir  || exit 1;
fi

if [ $stage -le 9 ]; then
  rm -f exp/$mic/nnet3/.error 2>/dev/null
  for data in dev eval; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 8 \
      data/$mic/${data}_hires exp/$mic/nnet3/extractor exp/$mic/nnet3/ivectors_${data} || touch exp/$mic/nnet3/.error &
  done
  wait
  [ -f exp/$mic/nnet3/.error ] && echo "$0: error extracting iVectors." && exit 1;
fi


if [ $stage -le 10 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for decode_set in dev eval; do
      (
      num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      decode_dir=${dir}/decode_${decode_set}

      steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
          --online-ivector-dir exp/$mic/nnet3/ivectors_${decode_set} \
         $graph_dir data/$mic/${decode_set}_hires $decode_dir || exit 1;
      ) &
  done
fi
wait;
exit 0;
