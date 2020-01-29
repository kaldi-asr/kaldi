#!/usr/bin/env bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a bottleneck feature extractor with 
# 'Universal Context' topology as invented by Frantisek Grezl,
# the network is on top of FBANK+f0 features.

. ./cmd.sh
. ./path.sh

# Config:
stage=0 # resume training with --stage=N
has_fisher=true
# End of config.
. utils/parse_options.sh || exit 1;
#

set -euxo pipefail 

train_src=data/train_nodup
train=data-fbank-pitch/train_nodup

dev_src=data/eval2000
dev=data-fbank-pitch/eval2000

gmmdir=exp/tri4

lang=data/lang
lang_test=data/lang_sw1_tg

if [ $stage -le 1 ]; then
  [ -e $dev ] && echo "Existing '$dev', better quit than overwrite!!!" && exit 1
  # prepare the FBANK+f0 features,
  # eval2000,
  utils/copy_data_dir.sh  $dev_src $dev; rm $dev/{feats,cmvn}.scp
  steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 $dev $dev/log $dev/data
  steps/compute_cmvn_stats.sh $dev $dev/log $dev/data
  # training set,
  utils/copy_data_dir.sh $train_src $train; rm $train/{feats,cmvn}.scp
  steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 $train $train/log $train/data
  steps/compute_cmvn_stats.sh $train $train/log $train/data
fi

if [ $stage -le 2 ]; then
  # split the data : 90% train, 10% cross-validation (held-out set),
  utils/subset_data_dir_tr_cv.sh $train ${train}_tr90 ${train}_cv10
fi

#########################################################################################
# Let's build universal-context bottleneck network
# - Universal context MLP is a hierarchy of two bottleneck neural networks
# - The first network has limited range of frames on input (11 frames)
# - The second network input is a concatenation of bottlneck outputs from the first 
#   network, with temporal shifts -10 -5..5 10, (in total a range of 31 frames 
#   in the original feature space)
# - This structure produces superior performance w.r.t. single bottleneck network
#
if [ $stage -le 3 ]; then
  # Train 1st network, overall context +/-5 frames
  # - the topology is 90_1500_1500_80_1500_NSTATES, linear bottleneck,
  dir=exp/nnet5uc-part1
  ali=${gmmdir}_ali_nodup
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --hid-layers 2 --hid-dim 1500 --bn-dim 80 \
      --cmvn-opts "--norm-means=true --norm-vars=false" \
      --feat-type traps --splice 5 --traps-dct-basis 6 --learn-rate 0.008 \
      ${train}_tr90 ${train}_cv10 $lang $ali $ali $dir
fi
#
if [ $stage -le 4 ]; then
  # Compose feature_transform for the next stage, 
  # - remaining part of the first network is fixed,
  dir=exp/nnet5uc-part1
  feature_transform=$dir/final.feature_transform.part1
  # Create splice transform,
  nnet-initialize <(echo "<Splice> <InputDim> 80 <OutputDim> 1040 <BuildVector> -10 -5:5 10 </BuildVector>") \
    $dir/splice_for_bottleneck.nnet 
  # Concatanate the input-transform, 1stage network, splicing,
  nnet-concat $dir/final.feature_transform "nnet-copy --remove-last-components=4 $dir/final.nnet - |" \
    $dir/splice_for_bottleneck.nnet $feature_transform
  
  # Train 2nd network, overall context +/-15 frames,
  # - the topology will be 1040_1500_1500_30_1500_NSTATES, linear bottleneck,
  # - cmvn_opts get imported inside 'train.sh',
  dir=exp/nnet5uc-part2
  ali=${gmmdir}_ali_nodup
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --hid-layers 2 --hid-dim 1500 --bn-dim 30 \
    --feature-transform $feature_transform --learn-rate 0.008 \
    ${train}_tr90 ${train}_cv10 $lang $ali $ali $dir
fi
#
#########################################################################################

# Decode the 2nd DNN,
if [ $stage -le 5 ]; then
  dir=exp/nnet5uc-part2
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.08333 \
    $gmmdir/graph_sw1_tg $dev $dir/decode_eval2000_sw1_tg
fi

# Store the BN-features,
nnet=exp/nnet5uc-part2
train_bn=data-$(basename $nnet)/train_nodup
dev_bn=data-$(basename $nnet)/eval2000
if [ $stage -le 6 ]; then
  # eval2000,
  steps/nnet/make_bn_feats.sh --cmd "$train_cmd" --nj 20 $dev_bn $dev $nnet $dev_bn/log $dev_bn/data
  # trainig,
  steps/nnet/make_bn_feats.sh --cmd "$train_cmd --max-jobs-run 50" --nj 200 $train_bn $train $nnet $train_bn/log $train_bn/data
  # For further GMM training, we have to produce cmvn statistics even if not used!!!
  steps/compute_cmvn_stats.sh $dev_bn $dev_bn/log $dev_bn/data
  steps/compute_cmvn_stats.sh $train_bn $train_bn/log $train_bn/data
fi

# Use single-pass retraining to build new GMM system on top of bottleneck features,
if [ $stage -le 7 ]; then
  dir=exp/tri6uc
  ali_src=${gmmdir}_ali_nodup
  graph=$dir/graph_${lang_test#*lang_}
  # Train,
  # GMM on bn features, no cmvn, no lda-mllt,
  steps/train_deltas.sh --cmd "$train_cmd" --delta-opts "--delta-order=0" \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --beam 20 --retry-beam 80 \
    11500 200000 $train_bn $lang $ali_src $dir 
  # Decode,
  utils/mkgraph.sh $lang_test $dir $graph
  steps/decode.sh --nj 30 --cmd "$decode_cmd" --acwt 0.05 --beam 15.0 --lattice-beam 8.0 \
    $graph $dev_bn $dir/decode_$(basename $dev_bn)_$(basename $graph)
  # Align,
  steps/align_fmllr.sh --nj 50 --cmd "$train_cmd" \
    --beam 20 --retry-beam 80 \
    $train_bn $lang $dir ${dir}_ali
fi

# Train SAT-adapted GMM on bottleneck features,
if [ $stage -le 8 ]; then
  dir=exp/tri7uc-sat
  ali=exp/tri6uc_ali
  graph=$dir/graph_${lang_test#*lang_}
  # Train,
  # fmllr-gmm system on bottleneck features, 
  # - no cmvn, put fmllr to the features directly (no lda),
  # - note1 : we don't need cmvn, similar effect has diagonal of fmllr transform,
  # - note2 : lda+mllt was causing a small hit <0.5%,
  steps/train_sat.sh --cmd "$train_cmd" --beam 20 --retry-beam 80 \
    11500 200000 $train_bn $lang $ali $dir
  # Decode,
  utils/mkgraph.sh $lang_test $dir $graph
  steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --acwt 0.05 --beam 15.0 --lattice-beam 8.0 \
    $graph $dev_bn $dir/decode_$(basename $dev_bn)_$(basename $graph)
fi

# Prepare alignments and lattices for bMMI training,
if [ $stage -le 9 ]; then
  dir=exp/tri7uc-sat
  # Align,
  steps/align_fmllr.sh --nj 50 --cmd "$train_cmd" --beam 20 --retry-beam 80 \
    $train_bn $lang $dir ${dir}_ali_nodup
  # Make denlats,
  steps/make_denlats.sh --nj 50 --cmd "$decode_cmd" --acwt 0.05 \
    --config conf/decode.config --transform-dir ${dir}_ali_nodup \
    $train_bn $lang $dir ${dir}_denlats_nodup 
fi

# 4 iterations of bMMI seems to work well overall. The number of iterations is
# used as an explicit argument even though train_mmi.sh will use 4 iterations by
# default.
num_mmi_iters=4
if [ $stage -le 10 ]; then
  dir=exp/tri7uc-sat_mmi_b0.1
  graph=exp/tri7uc-sat/graph_${lang_test#*lang_}
  steps/train_mmi.sh --cmd "$decode_cmd" \
    --boost 0.1 --num-iters $num_mmi_iters \
    $train_bn $lang exp/tri7uc-sat_{ali,denlats}_nodup ${dir}
  for iter in 1 2 3 4; do
    steps/decode.sh --nj 30 --cmd "$decode_cmd" --acwt 0.05 \
      --config conf/decode.config --iter $iter \
      --transform-dir exp/tri7uc-sat/decode_$(basename $dev_bn)_$(basename $graph) \
      $graph $dev_bn $dir/decode_$(basename $dev_bn)_$(basename $graph)_it${iter}
  done
fi

if [ $stage -le 11 ]; then
  if $has_fisher; then
    # Rescore with the 4gram swbd+fisher language model.
    dir=exp/tri7uc-sat_mmi_b0.1
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_sw1_{tg,fsh_fg} data/eval2000 \
      $dir/decode_eval2000_graph_sw1_{tg,fsh_fg}_it4
  fi
fi

echo Done.
