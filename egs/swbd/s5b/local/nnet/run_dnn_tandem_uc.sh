#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a bottleneck feature extractor with 
# 'Universal Context' topology as invented by Frantisek Grezl,
# the network is on top of FBANK+f0 features.

. cmd.sh
. path.sh

# Config:
stage=0 # resume training with --stage=N
# End of config.
. utils/parse_options.sh || exit 1;
#

if [ $stage -le 1 ]; then
  # prepare the FBANK+f0 features
  # eval2000
  dir=data-fbank-pitch/eval2000; srcdir=data/eval2000
  (mkdir -p $dir; cp $srcdir/* $dir; )
  steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 $dir $dir/log $dir/data || exit 1;
  steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;

  # training set
  dir=data-fbank-pitch/train; srcdir=data/train
  (mkdir -p $dir; cp $srcdir/* $dir; )
  steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 $dir $dir/log $dir/data || exit 1;
  steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
fi

if [ $stage -le 2 ]; then
  # Prepare same subsets as in the main MFCC-GMM recipe, these will be used 
  # during during building GMM system from flat-start, later in the Tandem recipe.
  data=data-fbank-pitch

  # Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
  utils/subset_data_dir.sh --first $data/train 4000 $data/train_dev # 5hr 6min
  n=$[`cat data/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last $data/train $n $data/train_nodev

  # Prepare data for training mono
  utils/subset_data_dir.sh --shortest $data/train_nodev 100000 $data/train_100kshort
  utils/subset_data_dir.sh  $data/train_100kshort 10000 $data/train_10k
  local/remove_dup_utts.sh 100 $data/train_10k $data/train_10k_nodup

  # Take the first 30k utterances (about 1/8th of the data)
  utils/subset_data_dir.sh --first $data/train_nodev 30000 $data/train_30k
  local/remove_dup_utts.sh 200 $data/train_30k $data/train_30k_nodup

  # Take the first 100k utterances (just under half the data); we'll use
  # this for later stages of training.
  utils/subset_data_dir.sh --first $data/train_nodev 100000 $data/train_100k
  local/remove_dup_utts.sh 200 $data/train_100k $data/train_100k_nodup

  # Full training dataset,
  local/remove_dup_utts.sh 300 $data/train_nodev $data/train_nodup
  # split the data : 90% train 10% cross-validation (held-out)
  dir=$data/train_nodup
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

#########################################################################################
# Let's build universal-context bottleneck network
# - Universal context MLP is a hierarchy of two bottleneck neural networks
# - The first network can see a limited range of frames (11 frames)
# - The second network sees concatenation of bottlneck outputs of the first 
#   network, with temporal shifts -10 -5 0 5 10, (in total a range of 31 frames 
#   in the original feature space)
# - This structure has been reported to produce superior performance
#   compared to a network with single bottleneck
#
if [ $stage -le 3 ]; then
  # 1st network, overall context +/-5 frames
  # - the topology is 90_1500_1500_80_1500_NSTATES, linear bottleneck
  dir=exp/nnet5b_uc-part1
  ali=exp/tri4b_ali_nodup
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --hid-layers 2 --hid-dim 1500 --bn-dim 80 --apply-cmvn true \
      --copy-feats false \
      --feat-type traps --splice 5 --traps-dct-basis 6 --learn-rate 0.008 \
    data-fbank-pitch/train_nodup_tr90 data-fbank-pitch/train_nodup_cv10 data/lang ${ali} ${ali} $dir || exit 1;
fi
if [ $stage -le 4 ]; then
  # Compose feature_transform for the next stage, 
  # - remaining part of the first network is fixed
  dir=exp/nnet5b_uc-part1
  feature_transform=$dir/final.feature_transform.part1
  nnet-concat $dir/final.feature_transform \
    "nnet-copy --remove-last-components=4 --binary=false $dir/final.nnet - |" \
    "utils/nnet/gen_splice.py --fea-dim=80 --splice=2 --splice-step=5 |" \
    $feature_transform || exit 1
  
  # 2nd network, overall context +/-15 frames
  # - the topology will be 400_1500_1500_30_1500_NSTATES, again, the bottleneck is linear
  dir=exp/nnet5b_uc-part2
  ali=exp/tri4b_ali_nodup
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --hid-layers 2 --hid-dim 1500 --bn-dim 30 --apply-cmvn true \
    --feature-transform $feature_transform --learn-rate 0.008 \
    data-fbank-pitch/train_nodup_tr90 data-fbank-pitch/train_nodup_cv10 data/lang ${ali} ${ali} $dir || exit 1;
fi
#
#########################################################################################

if [ $stage -le 5 ]; then
  # Store the BN-features
  data=data-bn/nnet5b_uc-part2 
  srcdata=data-fbank-pitch/
  nnet=exp/nnet5b_uc-part2
  # eval2000
  steps/nnet/make_bn_feats.sh --cmd "$train_cmd" --nj 20 $data/eval2000 $srcdata/eval2000 \
    $nnet $data/eval2000/log $data/eval2000/data || exit 1
  # trainig data (full set)
  steps/nnet/make_bn_feats.sh --cmd "$train_cmd" --nj 40 $data/train $srcdata/train \
    $nnet $data/train/log $data/train/data || exit 1

  # Compute CMVN of the BN-features
  dir=data-bn/nnet5b_uc-part2/train
  steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
  dir=data-bn/nnet5b_uc-part2/eval2000
  steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
fi

if [ $stage -le 6 ]; then
  # Prepare BN-feature subsets same as with MFCCs in run.sh 
  data=data-bn/nnet5b_uc-part2/

  # Use the first 4k sentences as dev set.
  utils/subset_data_dir.sh --first $data/train 4000 $data/train_dev # 5hr 6min
  n=$[`cat data/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last $data/train $n $data/train_nodev

  # Prepare data for training mono
  utils/subset_data_dir.sh --shortest $data/train_nodev 100000 $data/train_100kshort
  utils/subset_data_dir.sh  $data/train_100kshort 10000 $data/train_10k
  local/remove_dup_utts.sh 100 $data/train_10k $data/train_10k_nodup

  # Take the first 30k utterances (about 1/8th of the data)
  utils/subset_data_dir.sh --first $data/train_nodev 30000 $data/train_30k
  local/remove_dup_utts.sh 200 $data/train_30k $data/train_30k_nodup

  # Take the first 100k utterances (just under half the data); we'll use
  # this for later stages of training.
  utils/subset_data_dir.sh --first $data/train_nodev 100000 $data/train_100k
  local/remove_dup_utts.sh 200 $data/train_100k $data/train_100k_nodup

  # Full dataset
  local/remove_dup_utts.sh 300 $data/train_nodev $data/train_nodup
fi


# Start building the tandem GMM system
# - train from mono to tri4b, run bmmi training
bndata=data-bn/nnet5b_uc-part2/

if [ $stage -le 7 ]; then
  steps/tandem/train_mono.sh --nj 10 --cmd "$train_cmd" \
    data/train_10k_nodup $bndata/train_10k_nodup data/lang exp/tandem2uc-mono0a || exit 1;

  steps/tandem/align_si.sh --nj 30 --cmd "$train_cmd" \
     data/train_30k_nodup $bndata/train_30k_nodup data/lang exp/tandem2uc-mono0a exp/tandem2uc-mono0a_ali || exit 1;

  steps/tandem/train_deltas.sh --cmd "$train_cmd" \
      3200 30000 data/train_30k_nodup $bndata/train_30k_nodup data/lang exp/tandem2uc-mono0a_ali exp/tandem2uc-tri1 || exit 1;
   
  utils/mkgraph.sh data/lang_test exp/tandem2uc-tri1 exp/tandem2uc-tri1/graph

  steps/tandem/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode_tandem.config \
    exp/tandem2uc-tri1/graph data/eval2000 $bndata/eval2000 exp/tandem2uc-tri1/decode_eval2000
fi

if [ $stage -le 8 ]; then
  steps/tandem/align_si.sh --nj 30 --cmd "$train_cmd" \
     data/train_30k_nodup $bndata/train_30k_nodup data/lang exp/tandem2uc-tri1 exp/tandem2uc-tri1_ali || exit 1;

  steps/tandem/train_deltas.sh --cmd "$train_cmd" \
     3200 30000 data/train_30k_nodup $bndata/train_30k_nodup data/lang exp/tandem2uc-tri1_ali exp/tandem2uc-tri2 || exit 1;

  utils/mkgraph.sh data/lang_test exp/tandem2uc-tri2 exp/tandem2uc-tri2/graph || exit 1;
  steps/tandem/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode_tandem.config \
   exp/tandem2uc-tri2/graph data/eval2000 $bndata/eval2000 exp/tandem2uc-tri2/decode_eval2000 || exit 1;
fi

if [ $stage -le 9 ]; then
  steps/tandem/align_si.sh --nj 30 --cmd "$train_cmd" \
    data/train_100k_nodup $bndata/train_100k_nodup data/lang exp/tandem2uc-tri2 exp/tandem2uc-tri2_ali || exit 1;

  # Train tri3b, which is LDA+MLLT, on 100k_nodup data.
  steps/tandem/train_lda_mllt.sh --cmd "$train_cmd" \
     --splice-opts "--left-context=3 --right-context=3" \
     5500 90000 data/train_100k_nodup $bndata/train_100k_nodup data/lang exp/tandem2uc-tri2_ali exp/tandem2uc-tri3b || exit 1;

  utils/mkgraph.sh data/lang_test exp/tandem2uc-tri3b exp/tandem2uc-tri3b/graph || exit 1;
  steps/tandem/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode_tandem.config \
   exp/tandem2uc-tri3b/graph data/eval2000 $bndata/eval2000 exp/tandem2uc-tri3b/decode_eval2000 || exit 1;
fi

if [ $stage -le 10 ]; then
  # From now, we start building a more serious system (with SAT), 
  # and we'll do the alignment with fMLLR.
  steps/tandem/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train_nodup $bndata/train_nodup data/lang exp/tandem2uc-tri3b exp/tandem2uc-tri3b_ali_nodup || exit 1;

  steps/tandem/train_sat.sh  --cmd "$train_cmd" \
    11500 200000 data/train_nodup $bndata/train_nodup data/lang exp/tandem2uc-tri3b_ali_nodup exp/tandem2uc-tri4b || exit 1;

  utils/mkgraph.sh data/lang_test exp/tandem2uc-tri4b exp/tandem2uc-tri4b/graph || exit 1
  steps/tandem/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode_tandem.config \
    exp/tandem2uc-tri4b/graph data/eval2000 $bndata/eval2000 exp/tandem2uc-tri4b/decode_eval2000 || exit 1
fi

# bMMI starting from system in tandem2uc-tri4b, use full dataset.
if [ $stage -le 11 ]; then
  steps/tandem/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
    data/train_nodup $bndata/train_nodup data/lang exp/tandem2uc-tri4b exp/tandem2uc-tri4b_ali || exit 1;
  steps/tandem/make_denlats.sh --nj 40 --cmd "$decode_cmd" --transform-dir exp/tandem2uc-tri4b_ali \
    --sub-split 100 data/train_nodup $bndata/train_nodup data/lang exp/tandem2uc-tri4b exp/tandem2uc-tri4b_denlats || exit 1;
fi
if [ $stage -le 12 ]; then
  steps/tandem/train_mmi.sh --cmd "$decode_cmd" --boost 0.1 --acwt 0.039 \
    data/train_nodup $bndata/train_nodup data/lang exp/tandem2uc-tri4b_{ali,denlats} exp/tandem2uc-tri4b_mmi_b0.1 || exit 1;

  steps/tandem/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode_tandem.config \
    --transform-dir exp/tandem2uc-tri4b/decode_eval2000 \
    exp/tandem2uc-tri4b/graph data/eval2000 $bndata/eval2000 exp/tandem2uc-tri4b_mmi_b0.1/decode_eval2000 || exit 1;
fi

echo success
exit 0
