#!/usr/bin/env bash

# Copyright 2017   Nagendra Kumar Goel
#           2018   Vimal Manohar
#           2020   Desh Raj (Johns Hopkins University)
# Apache 2.0

# This is a script to train a TDNN-LSTM for overlap detections 
# using statistics pooling for long-context information.

stage=0
train_stage=-10
get_egs_stage=-10
egs_opts=

chunk_width=50

# The context is chosen to be around 1 second long. The context at test time
# is expected to be around the same.
extra_left_context=79
extra_right_context=21

relu_dim=512

# training options
num_epochs=40
initial_effective_lrate=0.00001
final_effective_lrate=0.000001
num_jobs_initial=8
num_jobs_final=12
remove_egs=true
max_param_change=0.2  # Small max-param change for small network

egs_dir=
nj=40

dir=
affix=1a

data_dir=
targets_dir=

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi
. ./utils/parse_options.sh

set -o pipefail
set -u

if [ -z "$dir" ]; then
  dir=exp/overlap_1a/tdnn_lstm
fi
dir=$dir${affix:+_$affix}

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

mkdir -p $dir

samples_per_iter=`perl -e "print int(400000 / $chunk_width)"`
cmvn_opts="--norm-means=false --norm-vars=false"
max_chunk_size=1000
echo $cmvn_opts > $dir/cmvn_opts

if [ $stage -le 1 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  tdnn_opts="l2-regularize=0.01"
  lstm_opts="l2-regularize=0.01 cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3"
  output_opts="l2-regularize=0.01"
  label_delay=5
  
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=`feat-to-dim scp:$data_dir/feats.scp -` name=input

  fixed-affine-layer name=lda input=Append(-1,0,1) affine-transform-file=$dir/configs/lda.mat
  
  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 $tdnn_opts dim=512
  relu-batchnorm-layer name=tdnn2 $tdnn_opts input=Append(-1,0,1) dim=512
  fast-lstmp-layer name=lstm1 $lstm_opts
  
  relu-batchnorm-layer name=tdnn3 $tdnn_opts input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn4 $tdnn_opts input=Append(-3,0,3) dim=512
  fast-lstmp-layer name=lstm2 $lstm_opts
  
  relu-batchnorm-layer name=tdnn5 $tdnn_opts input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn6 $tdnn_opts input=Append(-3,0,3) dim=512
  fast-lstmp-layer name=lstm3 $lstm_opts
  
  ## adding the layers for chain branch
  output-layer name=output input=lstm3 output-delay=$label_delay include-log-softmax=true dim=3

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig \
    --config-dir $dir/configs/

  cat <<EOF >> $dir/configs/vars
num_targets=3
EOF
fi

if [ $stage -le 2 ]; then
  num_utts=`cat $data_dir/utt2spk | wc -l`
  # Set num_utts_subset for diagnostics to a reasonable value
  # of max(min(0.005 * num_utts, 300), 12)
  num_utts_subset=`perl -e '$n=int($ARGV[0] * 0.005); print ($n > 300 ? 300 : ($n < 12 ? 12 : $n))' $num_utts`

  steps/nnet3/train_raw_rnn.py --stage=$train_stage \
    --feat.cmvn-opts="$cmvn_opts" \
    --egs.chunk-width=$chunk_width \
    --egs.dir="$egs_dir" --egs.stage=$get_egs_stage \
    --egs.chunk-left-context=$extra_left_context \
    --egs.chunk-right-context=$extra_right_context \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --trainer.num-epochs=$num_epochs \
    --trainer.samples-per-iter=20000 \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.rnn.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.5 \
    --trainer.deriv-truncate-margin=10 \
    --trainer.max-param-change=$max_param_change \
    --trainer.compute-per-dim-accuracy=true \
    --cmd="$decode_cmd" --nj $nj \
    --cleanup=true \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=wait \
    --use-dense-targets=true \
    --feat-dir=$data_dir \
    --targets-scp="$targets_dir/targets.scp" \
    --egs.opts="--frame-subsampling-factor 1 --num-utts-subset $num_utts_subset" \
    --dir=$dir || exit 1
fi

if [ $stage -le 3 ]; then
  # Use a subset to compute prior over the output targets
  $train_cmd $dir/log/get_priors.log \
   matrix-sum-rows scp:$targets_dir/targets.scp \
   ark:- \| vector-sum --binary=false ark:- $dir/post_output.vec || exit 1
  echo 1 > $dir/frame_subsampling_factor
fi
