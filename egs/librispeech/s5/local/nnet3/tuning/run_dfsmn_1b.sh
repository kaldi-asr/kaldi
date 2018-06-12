#!/bin/bash

# Copyright 2015  Johns Hopkins University (Author: Daniel Povey).
#           2018  Gaofeng Cheng (University of Chinese Academy of Sciences)
#           2018  Lu Huang (Tsinghua University)
# Apache 2.0.

# This is an example to train DFSMN (https://arxiv.org/pdf/1803.05030.pdf) under Kaldi Nnet3.
# What's special is that Semi-orthogonal (http://www.danielpovey.com/files/2018_interspeech_tdnnf.pdf) is applied to DFSMN,
# which is found beneficial to DFSMN.

# This is based on 1a, but a uniform l2 value is used.

# ./steps/info/nnet3_dir_info.pl exp/nnet3_cleaned/dfsmn_1a_sp/
# exp/nnet3_cleaned/dfsmn_1a_sp/: num-iters=818 nj=3..16 num-params=26.4M dim=40+100->5800 combine=-0.47->-0.47 (over 13) loglike:train/valid[544,817,combined]=(-0.52,-0.45,-0.44/-0.49,-0.46,-0.46) accuracy:train/valid[544,817,combined]=(0.826,0.845,0.846/0.835,0.846,0.844)

# ./steps/info/nnet3_dir_info.pl exp/nnet3_cleaned/dfsmn_1a_sp/
# exp/nnet3_cleaned/dfsmn_1b_sp/: num-iters=818 nj=3..16 num-params=26.4M dim=40+100->5800 combine=-0.46->-0.46 (over 6) loglike:train/valid[544,817,combined]=(-0.53,-0.46,-0.45/-0.51,-0.46,-0.46) accuracy:train/valid[544,817,combined]=(0.826,0.847,0.847/0.829,0.851,0.851)


# for test in dev_clean test_clean dev_other test_other; do for lm in fglarge tglarge tgmed tgsmall; do grep WER exp/nnet3_cleaned/dfsmn_1b_sp/decode_${test}_${lm}/wer* | best_wer.sh; done; echo; done

# %WER 4.06 [ 2207 / 54402, 304 ins, 215 del, 1688 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_dev_clean_fglarge/wer_12_0.5
# %WER 4.16 [ 2264 / 54402, 299 ins, 230 del, 1735 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_dev_clean_tglarge/wer_11_0.5
# %WER 5.13 [ 2789 / 54402, 343 ins, 286 del, 2160 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_dev_clean_tgmed/wer_12_0.0
# %WER 5.68 [ 3092 / 54402, 352 ins, 350 del, 2390 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_dev_clean_tgsmall/wer_12_0.0

# %WER 11.14 [ 5678 / 50948, 641 ins, 711 del, 4326 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_dev_other_fglarge/wer_14_0.5
# %WER 11.53 [ 5873 / 50948, 720 ins, 698 del, 4455 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_dev_other_tglarge/wer_15_0.0
# %WER 13.57 [ 6916 / 50948, 733 ins, 974 del, 5209 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_dev_other_tgmed/wer_15_0.0
# %WER 14.54 [ 7407 / 50948, 715 ins, 1105 del, 5587 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_dev_other_tgsmall/wer_15_0.0

# %WER 4.61 [ 2426 / 52576, 326 ins, 267 del, 1833 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_test_clean_fglarge/wer_12_1.0
# %WER 4.73 [ 2489 / 52576, 363 ins, 243 del, 1883 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_test_clean_tglarge/wer_11_0.5
# %WER 5.73 [ 3014 / 52576, 426 ins, 298 del, 2290 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_test_clean_tgmed/wer_12_0.0
# %WER 6.28 [ 3302 / 52576, 442 ins, 334 del, 2526 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_test_clean_tgsmall/wer_11_0.0

# %WER 11.40 [ 5969 / 52343, 750 ins, 650 del, 4569 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_test_other_fglarge/wer_13_0.0
# %WER 11.77 [ 6162 / 52343, 725 ins, 737 del, 4700 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_test_other_tglarge/wer_14_0.0
# %WER 13.77 [ 7210 / 52343, 744 ins, 983 del, 5483 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_test_other_tgmed/wer_14_0.0
# %WER 14.92 [ 7812 / 52343, 741 ins, 1132 del, 5939 sub ] exp/nnet3_cleaned/dfsmn_1b_sp/decode_test_other_tgsmall/wer_14_0.0

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=11
min_seg_len=1.55
train_set=train_960_cleaned
gmm=tri6b_cleaned
nnet3_affix=_cleaned

# Options which are not passed through to run_ivector_common.sh
train_stage=-10
affix=
common_egs_dir=
reporting_email=
remove_egs=true
egs_stage=-10
decode_nj=30

# training opts
chunk_width=80,40,20
chunk_left_context=0
chunk_right_context=0
num_epochs=12
num_jobs_initial=3
num_jobs_final=16
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
momentum=0.5
num_chunk_per_minibatch=100
samples_per_iter=20000


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


dir=exp/nnet3_cleaned/dfsmn_1b_sp
ali_dir=exp/${gmm}_ali_${train_set}_sp_comb
gmm_dir=exp/${gmm}
graph_dir=$gmm_dir/graph_tgsmall
train_data_dir=data/${train_set}_sp_hires_comb
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb


for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
     $graph_dir/HCLG.fst $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


local/nnet3/run_ivector_common.sh --stage $stage \
                                  --min-seg-len $min_seg_len \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --nnet3-affix "$nnet3_affix" || exit 1;

if [ $stage -le 11 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $ali_dir/tree |grep num-pdfs|awk '{print $2}')
  opts="l2-regularize=0.0007"
  linear_opts="orthonormal-constraint=-1.0 l2-regularize=0.0007"
  output_opts="l2-regularize=0.0007"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 input=Append(-1,0,1) dim=1536 $opts
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
  relu-batchnorm-layer name=prefinal-ce1 input=dfsmn9_projection dim=512 target-rms=0.5 $opts
  relu-batchnorm-layer name=prefinal-ce2 input=prefinal-ce1 dim=1536 target-rms=0.5 $opts
  output-layer name=output input=prefinal-ce2 dim=$num_targets max-change=1.5 $output_opts 

EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 12 ]; then
  steps/nnet3/train_rnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --feat.online-ivector-dir $train_ivector_dir \
    --trainer.num-epochs=$num_epochs \
    --trainer.samples-per-iter=$samples_per_iter \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.rnn.num-chunk-per-minibatch=$num_chunk_per_minibatch \
    --trainer.optimization.momentum=$momentum \
    --egs.stage=$egs_stage  \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.dir="$common_egs_dir" \
    --cleanup=true \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=100 \
    --use-gpu=true \
    --feat-dir=$train_data_dir \
    --ali-dir=$ali_dir \
    --lang=data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi


if [ $stage -le 13 ]; then
  if [ -z $extra_left_context ]; then
    extra_left_context=$chunk_left_context
  fi
  if [ -z $extra_right_context ]; then
    extra_right_context=$chunk_right_context
  fi
  if [ -z $frames_per_chunk ]; then
    frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  fi
  for decode_set in test_clean test_other dev_clean dev_other; do
    (
    steps/nnet3/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
      --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
      --extra-left-context $extra_left_context  \
      --extra-right-context $extra_right_context  \
      --frames-per-chunk "$frames_per_chunk" \
      $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_tgsmall || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/${decode_set}_hires $dir/decode_${decode_set}_{tgsmall,tgmed}  || exit 1
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/${decode_set}_hires $dir/decode_${decode_set}_{tgsmall,tglarge} || exit 1
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
      data/${decode_set}_hires $dir/decode_${decode_set}_{tgsmall,fglarge} || exit 1
    ) &
  done
fi
wait;
exit 0;
