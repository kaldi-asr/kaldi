#!/bin/bash

# 2016 Modified by Takafumi Moriya at Tokyo Institute of Technology
# for Japanese speech recognition using CSJ.

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a bottleneck feature extractor with 
# 'Universal Context' topology as invented by Frantisek Grezl,
# the network is on top of FBANK+f0 features.

. cmd.sh
. path.sh

# Config:
stage=0 # resume training with --stage=N
use_dev=false
# End of config.
. utils/parse_options.sh || exit 1;
#

[ ! -e data-fbank/train ] && if [ $stage -le 1 ]; then
    # prepare the FBANK+f0 features
    # all evaluation sets
    for eval_num in eval1 eval2 eval3 ;do
	dir=data-fbank/$eval_num; srcdir=data/$eval_num
	(mkdir -p $dir; cp $srcdir/* $dir; )
	steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 $dir $dir/log $dir/data || exit 1;
	steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
    done
    # training set
    dir=data-fbank/train; srcdir=data/train
    (mkdir -p $dir; cp $srcdir/* $dir; )
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 $dir $dir/log $dir/data || exit 1;
    steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
fi

if [ $stage -le 2 ]; then
  # Prepare same subsets as in the main MFCC-GMM recipe, these will be used 
  # during during building GMM system from flat-start, later in the Tandem recipe.
  data=data-fbank

  if $use_dev ;then
    dev_set=train_dev
  # Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
    utils/subset_data_dir.sh --first $data/train 4000 $data/$dev_set # 6hr 31min
    n=$[`cat data/train/segments | wc -l` - 4000]
    utils/subset_data_dir.sh --last $data/train $n $data/train_nodev
  else
    cp -r $data/train $data/train_nodev
  fi
  
  # Prepare data for training mono
  # Take the first 30k utterances (about 1/8th of the data)
  utils/subset_data_dir.sh --shortest $data/train_nodev 100000 $data/train_100kshort
  utils/subset_data_dir.sh $data/train_100kshort 30000 $data/train_30kshort

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
  ali=exp/tri4_ali_nodup
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --hid-layers 2 --hid-dim 1500 --bn-dim 80 --apply-cmvn true \
      --copy-feats false \
      --feat-type traps --splice 5 --traps-dct-basis 6 --learn-rate 0.008 \
    data-fbank/train_nodup_tr90 data-fbank/train_nodup_cv10 data/lang ${ali} ${ali} $dir || exit 1;
fi
if [ $stage -le 4 ]; then
  # Compose feature_transform for the next stage, 
  # - remaining part of the first network is fixed
  dir=exp/nnet5b_uc-part1
  feature_transform=$dir/final.feature_transform.part1
  nnet-concat $dir/final.feature_transform \
    "nnet-copy --remove-last-layers=4 --binary=false $dir/final.nnet - |" \
    "utils/nnet/gen_splice.py --fea-dim=80 --splice=2 --splice-step=5 |" \
    $feature_transform || exit 1
  
  # 2nd network, overall context +/-15 frames
  # - the topology will be 400_1500_1500_30_1500_NSTATES, again, the bottleneck is linear
  dir=exp/nnet5b_uc-part2
  ali=exp/tri4_ali_nodup
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --hid-layers 2 --hid-dim 1500 --bn-dim 30 --apply-cmvn true \
    --feature-transform $feature_transform --learn-rate 0.008 \
    data-fbank/train_nodup_tr90 data-fbank/train_nodup_cv10 data/lang ${ali} ${ali} $dir || exit 1;
fi
#
#########################################################################################

if [ $stage -le 5 ]; then
  # Store the BN-features
  data=data-bn/nnet5b_uc-part2 
  srcdata=data-fbank
  nnet=exp/nnet5b_uc-part2
  
  # all evaluation sets
  for eval_num in eval1 eval2 eval3 $dev_set ;do
      steps/nnet/make_bn_feats.sh --cmd "$train_cmd" --nj 10 $data/$eval_num $srcdata/$eval_num \
	  $nnet $data/$eval_num/log $data/$eval_num/data || exit 1
  done
  # trainig data (full set)
  steps/nnet/make_bn_feats.sh --cmd "$train_cmd" --nj 10 $data/train $srcdata/train \
    $nnet $data/train/log $data/train/data || exit 1

  # Compute CMVN of the BN-features
  dir=data-bn/nnet5b_uc-part2/train
  steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;

  for eval_num in eval1 eval2 eval3 $dev_set ;do
      dir=data-bn/nnet5b_uc-part2/$eval_num
      steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
  done
fi

if [ $stage -le 6 ]; then
  # Prepare BN-feature subsets same as with MFCCs in run.sh 
  data=data-bn/nnet5b_uc-part2
  
  if $use_dev ;then
      dev_set=train_dev
      # Use the first 4k sentences as dev set.
      utils/subset_data_dir.sh --first $data/train 4000 $data/$dev_set # 6hr 31min
      n=$[`cat data/train/segments | wc -l` - 4000]
      utils/subset_data_dir.sh --last $data/train $n $data/train_nodev
  else
      cp -r $data/train $data/train_nodev
  fi

  # Prepare data for training mono
  # Take the first 30k utterances (about 1/8th of the data)
  utils/subset_data_dir.sh --shortest $data/train_nodev 100000 $data/train_100kshort
  utils/subset_data_dir.sh $data/train_100kshort 30000 $data/train_30kshort

  # Take the first 100k utterances (just under half the data); we'll use
  # this for later stages of training.
  utils/subset_data_dir.sh --first $data/train_nodev 100000 $data/train_100k
  local/remove_dup_utts.sh 200 $data/train_100k $data/train_100k_nodup

  # Full dataset
  local/remove_dup_utts.sh 300 $data/train_nodev $data/train_nodup
fi


# Start building the tandem GMM system
# - train from mono to tri4, run bmmi training
bndata=data-bn/nnet5b_uc-part2

if [ $stage -le 7 ]; then
  steps/tandem/train_mono.sh --nj 10 --cmd "$train_cmd" \
    data/train_30kshort $bndata/train_30kshort data/lang exp/tandem2uc-mono0a || exit 1;

  steps/tandem/align_si.sh --nj 10 --cmd "$train_cmd" \
     data/train_100k_nodup $bndata/train_100k_nodup data/lang exp/tandem2uc-mono0a exp/tandem2uc-mono0a_ali || exit 1;

  steps/tandem/train_deltas.sh --cmd "$train_cmd" \
      3200 30000 data/train_100k_nodup $bndata/train_100k_nodup data/lang exp/tandem2uc-mono0a_ali exp/tandem2uc-tri1 || exit 1;
   
  utils/mkgraph.sh data/lang_csj_tg exp/tandem2uc-tri1 exp/tandem2uc-tri1/graph_csj_tg

  for eval_num in eval1 eval2 eval3 $dev_set ;do
      steps/tandem/decode.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_tandem.config \
	  exp/tandem2uc-tri1/graph_csj_tg data/$eval_num $bndata/$eval_num exp/tandem2uc-tri1/decode_${eval_num}_csj
  done
fi

if [ $stage -le 8 ]; then
  steps/tandem/align_si.sh --nj 10 --cmd "$train_cmd" \
     data/train_100k_nodup $bndata/train_100k_nodup data/lang exp/tandem2uc-tri1 exp/tandem2uc-tri1_ali || exit 1;

  steps/tandem/train_deltas.sh --cmd "$train_cmd" \
     4000 70000 data/train_100k_nodup $bndata/train_100k_nodup data/lang exp/tandem2uc-tri1_ali exp/tandem2uc-tri2 || exit 1;

  utils/mkgraph.sh data/lang_csj_tg exp/tandem2uc-tri2 exp/tandem2uc-tri2/graph_csj_tg || exit 1;
  for eval_num in eval1 eval2 eval3 $dev_set ;do
      steps/tandem/decode.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_tandem.config \
	  exp/tandem2uc-tri2/graph_csj_tg data/$eval_num $bndata/$eval_num exp/tandem2uc-tri2/decode_${eval_num}_csj || exit 1;
  done
fi

if [ $stage -le 9 ]; then
  steps/tandem/align_si.sh --nj 10 --cmd "$train_cmd" \
    data/train_nodup $bndata/train_nodup data/lang exp/tandem2uc-tri2 exp/tandem2uc-tri2_ali || exit 1;

  # Train tri3, which is LDA+MLLT, on train_nodup data.
  steps/tandem/train_lda_mllt.sh --cmd "$train_cmd" \
     --splice-opts "--left-context=3 --right-context=3" \
     6000 140000 data/train_nodup $bndata/train_nodup data/lang exp/tandem2uc-tri2_ali exp/tandem2uc-tri3 || exit 1;

  utils/mkgraph.sh data/lang_csj_tg exp/tandem2uc-tri3 exp/tandem2uc-tri3/graph_csj_tg || exit 1;
  for eval_num in eval1 eval2 eval3 $dev_set ;do
      steps/tandem/decode.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_tandem.config \
	  exp/tandem2uc-tri3/graph_csj_tg data/$eval_num $bndata/$eval_num exp/tandem2uc-tri3/decode_${eval_num}_csj || exit 1;
  done
fi

if [ $stage -le 10 ]; then
  # From now, we start building a more serious system (with SAT), 
  # and we'll do the alignment with fMLLR.
  steps/tandem/align_fmllr.sh --nj 10 --cmd "$train_cmd" \
    data/train_nodup $bndata/train_nodup data/lang exp/tandem2uc-tri3 exp/tandem2uc-tri3_ali_nodup || exit 1;

  steps/tandem/train_sat.sh  --cmd "$train_cmd" \
    11500 200000 data/train_nodup $bndata/train_nodup data/lang exp/tandem2uc-tri3_ali_nodup exp/tandem2uc-tri4 || exit 1;

  utils/mkgraph.sh data/lang_csj_tg exp/tandem2uc-tri4 exp/tandem2uc-tri4/graph_csj_tg || exit 1
  for eval_num in eval1 eval2 eval3 $dev_set ;do
      steps/tandem/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_tandem.config \
	  exp/tandem2uc-tri4/graph_csj_tg data/$eval_num $bndata/$eval_num exp/tandem2uc-tri4/decode_${eval_num}_csj || exit 1
  done
fi

# bMMI starting from system in tandem2uc-tri4, use full dataset.
if [ $stage -le 11 ]; then
  steps/tandem/align_fmllr.sh --nj 10 --cmd "$train_cmd" \
    data/train_nodup $bndata/train_nodup data/lang exp/tandem2uc-tri4 exp/tandem2uc-tri4_ali || exit 1;
  steps/tandem/make_denlats.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tandem2uc-tri4_ali \
    --sub-split 100 data/train_nodup $bndata/train_nodup data/lang exp/tandem2uc-tri4 exp/tandem2uc-tri4_denlats || exit 1;
fi
if [ $stage -le 12 ]; then
  steps/tandem/train_mmi.sh --cmd "$decode_cmd" --boost 0.1 --acwt 0.039 \
    data/train_nodup $bndata/train_nodup data/lang exp/tandem2uc-tri4_{ali,denlats} exp/tandem2uc-tri4_mmi_b0.1 || exit 1;

  for eval_num in eval1 eval2 eval3 $dev_set ;do
      steps/tandem/decode.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_tandem.config \
	  --transform-dir exp/tandem2uc-tri4/decode_${eval_num}_csj \
	  exp/tandem2uc-tri4/graph_csj_tg data/$eval_num $bndata/$eval_num exp/tandem2uc-tri4_mmi_b0.1/decode_${eval_num}_csj || exit 1;
  done
fi

echo success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
# We use config parameters of swbd resipe.
:<<EOF
=== evaluation set 1 ===
%WER 16.46 [ 4285 / 26028, 556 ins, 910 del, 2819 sub ] exp/tandem2uc-tri1/decode_eval1_csj/wer_20_0.0
%WER 15.32 [ 3987 / 26028, 523 ins, 825 del, 2639 sub ] exp/tandem2uc-tri2/decode_eval1_csj/wer_20_0.0
%WER 14.28 [ 3718 / 26028, 475 ins, 744 del, 2499 sub ] exp/tandem2uc-tri3/decode_eval1_csj/wer_20_0.0
%WER 13.51 [ 3517 / 26028, 450 ins, 738 del, 2329 sub ] exp/tandem2uc-tri4/decode_eval1_csj/wer_20_0.5
%WER 14.93 [ 3885 / 26028, 584 ins, 711 del, 2590 sub ] exp/tandem2uc-tri4/decode_eval1_csj.si/wer_20_0.0
%WER 12.42 [ 3232 / 26028, 399 ins, 671 del, 2162 sub ] exp/tandem2uc-tri4_mmi_b0.1/decode_eval1_csj/wer_20_1.0
=== evaluation set 2 ===
%WER 12.54 [ 3343 / 26661, 474 ins, 525 del, 2344 sub ] exp/tandem2uc-tri1/decode_eval2_csj/wer_20_0.0
%WER 12.19 [ 3250 / 26661, 371 ins, 596 del, 2283 sub ] exp/tandem2uc-tri2/decode_eval2_csj/wer_20_1.0
%WER 11.19 [ 2984 / 26661, 354 ins, 511 del, 2119 sub ] exp/tandem2uc-tri3/decode_eval2_csj/wer_20_0.5
%WER 9.96 [ 2655 / 26661, 349 ins, 427 del, 1879 sub ] exp/tandem2uc-tri4/decode_eval2_csj/wer_20_0.5
%WER 11.96 [ 3188 / 26661, 504 ins, 427 del, 2257 sub ] exp/tandem2uc-tri4/decode_eval2_csj.si/wer_20_0.0
%WER 9.30 [ 2480 / 26661, 312 ins, 387 del, 1781 sub ] exp/tandem2uc-tri4_mmi_b0.1/decode_eval2_csj/wer_20_1.0
=== evaluation set 3 ===
%WER 18.19 [ 3127 / 17189, 555 ins, 510 del, 2062 sub ] exp/tandem2uc-tri1/decode_eval3_csj/wer_20_0.5
%WER 17.80 [ 3060 / 17189, 522 ins, 535 del, 2003 sub ] exp/tandem2uc-tri2/decode_eval3_csj/wer_20_1.0
%WER 15.88 [ 2729 / 17189, 520 ins, 423 del, 1786 sub ] exp/tandem2uc-tri3/decode_eval3_csj/wer_20_0.5
%WER 14.88 [ 2557 / 17189, 556 ins, 359 del, 1642 sub ] exp/tandem2uc-tri4/decode_eval3_csj/wer_20_0.5
%WER 17.03 [ 2927 / 17189, 592 ins, 417 del, 1918 sub ] exp/tandem2uc-tri4/decode_eval3_csj.si/wer_20_1.0
%WER 13.44 [ 2311 / 17189, 430 ins, 340 del, 1541 sub ] exp/tandem2uc-tri4_mmi_b0.1/decode_eval3_csj/wer_20_1.0
EOF