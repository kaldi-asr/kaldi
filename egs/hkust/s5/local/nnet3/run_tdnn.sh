#!/bin/bash

# This script is based on swbd/s5c/local/nnet3/run_tdnn.sh

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.
set -e

stage=0
train_stage=-10
affix=
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

dir=exp/nnet3/tdnn_sp${affix:+_$affix}
gmm_dir=exp/tri5a
train_set=train_sp
ali_dir=${gmm_dir}_sp_ali
graph_dir=$gmm_dir/graph

local/nnet3/run_ivector_common.sh --stage $stage \
  --ivector-extractor exp/nnet2_online/extractor || exit 1;

if [ $stage -le 8 ]; then
  echo "$0: creating neural net configs";

  # create the config files for nnet initialization
  python steps/nnet3/tdnn/make_configs.py  \
    --feat-dir data/${train_set}_hires \
    --ivector-dir exp/nnet3/ivectors_${train_set} \
    --ali-dir $ali_dir \
    --relu-dim 850 \
    --splice-indexes "-2,-1,0,1,2 -1,2 -3,3 -7,2 -3,3 0 0"  \
    --use-presoftmax-prior-scale true \
   $dir/configs || exit 1;
fi

if [ $stage -le 9 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/hkust-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
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

if [ $stage -le 10 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for decode_set in dev; do
      (
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      decode_dir=${dir}/decode
      ivector_opts=" --online-ivector-dir exp/nnet3/ivectors_${decode_set} "

      steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" $ivector_opts \
         $graph_dir data/${decode_set}_hires $decode_dir || exit 1;
      ) &
  done
fi

if [ $stage -le 11 ]; then
  steps/online/nnet3/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    --add-pitch true \
    data/lang exp/nnet2_online/extractor "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 12 ]; then
  # do the actual online decoding with iVectors, carrying info forward from
  # previous utterances of the same speaker.
  graph_dir=exp/tri5a/graph
  steps/online/nnet3/decode.sh --config conf/decode.config \
    --cmd "$decode_cmd" --nj 10 \
    "$graph_dir" data/dev_hires \
    ${dir}_online/decode || exit 1;
fi

if [ $stage -le 13 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  graph_dir=exp/tri5a/graph
  steps/online/nnet3/decode.sh --config conf/decode.config \
    --cmd "$decode_cmd" --nj 10 --per-utt true \
    "$graph_dir" data/dev_hires \
    ${dir}_online/decode_per_utt || exit 1;
fi

wait;
exit 0;
