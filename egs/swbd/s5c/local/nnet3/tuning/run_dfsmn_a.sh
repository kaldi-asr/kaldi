#!/bin/bash

# Copyright 2015  Johns Hopkins University (Author: Daniel Povey).
#                 Gaofeng Cheng (University of Chinese Academy of Sciences)
# Apache 2.0.

# This is an example to train DFSMN (https://arxiv.org/pdf/1803.05030.pdf) under Kaldi Nnet3.
# What's special is that Semi-orthogonal (http://www.danielpovey.com/files/2018_interspeech_tdnnf.pdf) is applied to DFSMN,
# which is found to be beneficial to DFSMN.

# Results:
# System                dfsmn_1a_context_57_57_sp
# WER on train_dev(tg)      14.19
# WER on train_dev(fg)      13.25
# WER on eval2000(tg)        16.9
# WER on eval2000_swbd(tg)        11.5
# WER on eval2000_callhm(tg)        22.2
# WER on eval2000(fg)        15.7
# WER on eval2000_swbd(fg)        10.4
# WER on eval2000_callhm(fg)        21.0
# Final train prob         -0.728
# Final valid prob         -0.838
# Final train accuracy          0.767
# Final valid accuracy          0.749

# Trying broader context and string-style chunk.
# Also. I'm setting up librispeech.
stage=9
affix=
train_stage=-10
has_fisher=true
speed_perturb=true
common_egs_dir=
reporting_email=
remove_egs=true
# training opts
label_delay=0
chunk_width=20
chunk_left_context=0
chunk_right_context=0
num_epochs=8
num_jobs_initial=3
num_jobs_final=15
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
momentum=0.5
num_chunk_per_minibatch=100
samples_per_iter=20000
#exp/nnet3/tdnn_c_sp/egs


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

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi
dir=exp/nnet3/dfsmn_1a
dir=$dir${affix:+_$affix}
dir=${dir}$suffix
train_set=train_nodup$suffix
ali_dir=exp/tri4_ali_nodup$suffix

local/nnet3/run_ivector_common.sh --stage $stage \
    --speed-perturb $speed_perturb || exit 1;

if [ $stage -le 9 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $ali_dir/tree |grep num-pdfs|awk '{print $2}')
  opts="l2-regularize=0.0007"
  linear_opts="orthonormal-constraint=-1.0 l2-regularize=0.0007"
  output_opts="l2-regularize=0.0007"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=60 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 input=Append(-1,0,1) dim=1536
  linear-component name=tdnn1l dim=512 $linear_opts
  
  blocksum-layer name=dfsmn1_blocksum input=Append(-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6) dim=512
  relu-batchnorm-layer name=dfsmn1_inter dim=1536  input=Sum(dfsmn1_blocksum, tdnn1l) $opts
  linear-component name=dfsmn1_projection dim=512  $linear_opts

  blocksum-layer name=dfsmn2_blocksum input=Append(-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6) dim=512
  relu-batchnorm-layer name=dfsmn2_inter dim=1536  input=Sum(dfsmn2_blocksum, dfsmn1_blocksum, dfsmn1_projection) $opts
  linear-component name=dfsmn2_projection dim=512  $linear_opts

  blocksum-layer name=dfsmn3_blocksum input=Append(-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6) dim=512
  relu-batchnorm-layer name=dfsmn3_inter dim=1536  input=Sum(dfsmn3_blocksum, dfsmn2_blocksum, dfsmn2_projection) $opts
  linear-component name=dfsmn3_projection dim=512  $linear_opts

  blocksum-layer name=dfsmn4_blocksum input=Append(-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6) dim=512
  relu-batchnorm-layer name=dfsmn4_inter dim=1536  input=Sum(dfsmn4_blocksum, dfsmn3_blocksum, dfsmn3_projection) $opts
  linear-component name=dfsmn4_projection dim=512  $linear_opts

  blocksum-layer name=dfsmn5_blocksum input=Append(-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6) dim=512
  relu-batchnorm-layer name=dfsmn5_inter dim=1536  input=Sum(dfsmn5_blocksum, dfsmn4_blocksum, dfsmn4_projection) $opts
  linear-component name=dfsmn5_projection dim=512  $linear_opts

  blocksum-layer name=dfsmn6_blocksum input=Append(-6,-4,-2,0,2,4,6) dim=512
  relu-batchnorm-layer name=dfsmn6_inter dim=1536  input=Sum(dfsmn6_blocksum, dfsmn5_blocksum, dfsmn5_projection) $opts
  linear-component name=dfsmn6_projection dim=512  $linear_opts

  blocksum-layer name=dfsmn7_blocksum input=Append(-6,-4,-2,0,2,4,6) dim=512
  relu-batchnorm-layer name=dfsmn7_inter dim=1536  input=Sum(dfsmn7_blocksum, dfsmn6_blocksum, dfsmn6_projection) $opts
  linear-component name=dfsmn7_projection dim=512  $linear_opts

  blocksum-layer name=dfsmn8_blocksum input=Append(-6,-4,-2,0,2,4,6) dim=512
  relu-batchnorm-layer name=dfsmn8_inter dim=1536  input=Sum(dfsmn8_blocksum, dfsmn7_blocksum, dfsmn7_projection) $opts
  linear-component name=dfsmn8_projection dim=512  $linear_opts

  blocksum-layer name=dfsmn9_blocksum input=Append(-6,-4,-2,0,2,4,6) dim=512
  relu-batchnorm-layer name=dfsmn9_inter dim=1536  input=Sum(dfsmn9_blocksum, dfsmn8_blocksum, dfsmn8_projection) $opts
  linear-component name=dfsmn9_projection dim=512  $linear_opts

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-ce1 input=dfsmn9_projection dim=512 target-rms=0.5 l2-regularize=0.0015 
  relu-batchnorm-layer name=prefinal-ce2 input=prefinal-ce1 dim=1536 target-rms=0.5 $opts
  output-layer name=output input=prefinal-ce2 dim=$num_targets max-change=1.5 $output_opts 

EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 10 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_rnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --trainer.num-epochs=$num_epochs \
    --trainer.samples-per-iter=$samples_per_iter \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.rnn.num-chunk-per-minibatch=$num_chunk_per_minibatch \
    --trainer.optimization.momentum=$momentum \
    --egs.stage=-10  \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.dir="$common_egs_dir" \
    --cleanup=false \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=20 \
    --use-gpu=true \
    --feat-dir=data/${train_set}_hires \
    --ali-dir=$ali_dir \
    --lang=data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi


graph_dir=exp/tri4/graph_sw1_tg
if [ $stage -le 11 ]; then
  if [ -z $extra_left_context ]; then
    extra_left_context=$chunk_left_context
  fi
  if [ -z $extra_right_context ]; then
    extra_right_context=$chunk_right_context
  fi
  if [ -z $frames_per_chunk ]; then
    frames_per_chunk=$chunk_width
  fi
  for decode_set in train_dev eval2000 rt03; do
    (
    num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
      --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
      --extra-left-context $extra_left_context  \
      --extra-right-context $extra_right_context  \
      --frames-per-chunk "$frames_per_chunk" \
      $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_hires_sw1_tg || exit 1;
    if $has_fisher; then
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
        $dir/decode_${decode_set}_hires_sw1_{tg,fsh_fg} || exit 1;
    fi
    ) &
  done
fi
wait;
exit 0;

