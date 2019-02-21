#!/bin/bash

# This script is based on aishell/s5/local/nnet3/tuning/run_tdnn_1a.sh

# In this script, the neural network in trained based on hires mfcc and online pitch.
# The online pitch setup requires a online_pitch.conf in the conf dir for both training
# and testing.

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

dir=exp/nnet3/tdnn_sp${affix:+_$affix}
gmm_dir=exp/tri5a
train_set=train_sp
ali_dir=${gmm_dir}_sp_ali
graph_dir=$gmm_dir/graph

local/nnet3/run_ivector_common.sh --stage $stage --online true || exit 1;

if [ $stage -le 7 ]; then
  echo "$0: creating neural net configs";

  num_targets=$(tree-info $ali_dir/tree |grep num-pdfs|awk '{print $2}')

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=43 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=850
  relu-batchnorm-layer name=tdnn2 dim=850 input=Append(-1,0,2)
  relu-batchnorm-layer name=tdnn3 dim=850 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn4 dim=850 input=Append(-7,0,2)
  relu-batchnorm-layer name=tdnn5 dim=850 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn6 dim=850
  output-layer name=output input=tdnn6 dim=$num_targets max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aishell-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
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
    --feat-dir=data/${train_set}_hires_online \
    --ali-dir $ali_dir \
    --lang data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for decode_set in dev test; do
    num_jobs=`cat data/${decode_set}_hires_online/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    decode_dir=${dir}/decode_$decode_set
    steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
       --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
       $graph_dir data/${decode_set}_hires_online $decode_dir || exit 1;
  done
fi

if [ $stage -le 10 ]; then
  steps/online/nnet3/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    --add-pitch true \
    data/lang exp/nnet3/extractor "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 11 ]; then
  # do the actual online decoding with iVectors, carrying info forward from
  # previous utterances of the same speaker.
  for decode_set in dev test; do
    num_jobs=`cat data/${decode_set}_hires_online/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    decode_dir=${dir}_online/decode_$decode_set
    steps/online/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
       --config conf/decode.config \
       $graph_dir data/${decode_set}_hires_online $decode_dir || exit 1;
  done
fi

if [ $stage -le 12 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for decode_set in dev test; do
    num_jobs=`cat data/${decode_set}_hires_online/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    decode_dir=${dir}_online/decode_${decode_set}_per_utt
    steps/online/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
       --config conf/decode.config --per-utt true \
       $graph_dir data/${decode_set}_hires_online $decode_dir || exit 1;
  done
fi

wait;
exit 0;
