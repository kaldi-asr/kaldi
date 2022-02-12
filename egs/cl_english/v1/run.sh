#!/usr/bin/env bash

# Copyright 2021  Behavox (author: Hossein Hadian)
# Apache 2.0

# This script demonstrates one example of continual learning (CL)
# using Learning Without Forgetting (LWF) and DenLWF methods.
# This specific example is for transferring from a subset of Fisher English
# data to CommonVoice Indian subset. Therefore it's an example of accent adaptation
# using CL from American to Indian English.

# See this paper for more info: https://arxiv.org/abs/2110.07055

# To run this script, first prepare the Fisher train/test data under
# data/fsh_train and data/fsh_dev and LM under data/local/lm.gz.
# Also prepare the lang and dict under data/lang_nosp and
# data/local/dict_nosp respectively. Additionally, prepare the indian subset of
# Common Voice under data/cvi_train and data/cvi_test.
# Finally run these commands one by one:

run_baseline=false  # If set, will run LWF baseline for comparison.

# Note: You can change 10000 to 50000 to replicate the results in the paper.
utils/subset_data_dir.sh data/fsh_train 10000 data/fsh_train_10hr
utils/subset_data_dir.sh data/cvi_train 10000 data/cvi_train_10hr

utils/combine_data_dir.sh data/comb_train data/cvi_train_10hr data/fsh_train_10hr

# These train an standalone model for each dataset and the combined data:
# Note that we use mono tree for the source model so that LWF can be applied efficiently
local/run_pipeline.sh --softmax true --tree-affix mono --tree-opts \
                      "--context-width=1 --central-position=0"  cvi_train_10hr
local/run_pipeline.sh --softmax true --tree-affix mono --tree-opts \
                      "--context-width=1 --central-position=0"  fsh_train_10hr
local/run_pipeline.sh data/comb_train


# Do simple fine-tuning (i.e. transfer learning)
local/chain/run_finetune_1b.sh \
  --src-tree-dir exp_fsh_train_10hr/chain/tree_sp_mono \
  --dir exp_cvi_train_10hr/chain/tdnn_ft1b_fsh2cvi_sp \
  --exp exp_cvi_train_10hr --train-set cvi_train_10hr \
  --src-mdl exp_fsh_train_10hr/chain/tdnn1a_noiv_small_sp/final.mdl

# Do DenLWF continual learning (recommended method)
local/chain/run_lwf_clean_1a.sh \
  --src-tree-dir exp_fsh_train_10hr/chain/tree_sp_mono \
  --dir exp_cvi_train_10hr/chain/tdnn_denlwf1a_fsh2cvi_sp \
  --exp exp_cvi_train_10hr --train-set cvi_train_10hr \
  --src-mdl exp_fsh_train_10hr/chain/tdnn1a_noiv_small_sp/final.mdl \
  --lwf-den-scale 0.6

local/run_evaluation.sh --test-sets "cvi_test fsh_dev" exp_cvi_train_10hr/chain/tdnn_ft1b_fsh2cvi_sp/
local/run_evaluation.sh --test-sets "cvi_test fsh_dev" exp_cvi_train_10hr/chain/tdnn_denlwf1a_fsh2cvi_sp/



if $run_baseline; then
  # Do LWF continual learing (for comparison)
  local/chain/run_lwf_clean_1a.sh \
    --src-tree-dir exp_fsh_train_10hr/chain/tree_sp_mono \
    --dir exp_cvi_train_10hr/chain/tdnn_lwf1a_fsh2cvi_sp \
    --exp exp_cvi_train_10hr --train-set cvi_train_10hr \
    --src-mdl exp_fsh_train_10hr/chain/tdnn1a_noiv_small_sp/final.mdl \
    --lwf-scale 0.8 --lwf-den-scale ""

  local/run_evaluation.sh --test-sets "cvi_test fsh_dev" exp_cvi_train_10hr/chain/tdnn_lwf1a_fsh2cvi_sp/
fi


### Results:

# Standalone Fisher:
# %WER 76.58 [ 7566 / 9880, 374 ins, 2039 del, 5153 sub ] exp_fsh_train_10hr/chain/tdnn1a_noiv_small_sp/decode_graph_cvi_test_iterfinal/wer_15_1.0
# %WER 33.64 [ 13185 / 39195, 1347 ins, 3330 del, 8508 sub ] exp_fsh_train_10hr/chain/tdnn1a_noiv_small_sp/decode_graph_fsh_dev_iterfinal/wer_12_0.0

# Standalone CVI
# %WER 37.62 [ 3717 / 9880, 368 ins, 663 del, 2686 sub ] exp_cvi_train_10hr/chain/tdnn1a_noiv_small_sp/decode_graph_cvi_test_iterfinal/wer_9_0.0
# %WER 80.07 [ 31382 / 39195, 751 ins, 11945 del, 18686 sub ] exp_cvi_train_10hr/chain/tdnn1a_noiv_small_sp/decode_graph_fsh_dev_iterfinal/wer_9_0.0

# Simple FT (FSH --> CVI):
# %WER 42.61 [ 4210 / 9880, 350 ins, 1024 del, 2836 sub ] exp_cvi_train_10hr/chain/tdnn_ft1b_fsh2cvi_sp//decode_graph_cvi_test_iterfinal/wer_7_0.0
# %WER 79.84 [ 31294 / 39195, 399 ins, 16563 del, 14332 sub ] exp_cvi_train_10hr/chain/tdnn_ft1b_fsh2cvi_sp//decode_graph_fsh_dev_iterfinal/wer_7_0.0

# LWF (FSH --> CVI)
# %WER 49.64 [ 4904 / 9880, 391 ins, 1281 del, 3232 sub ] exp_cvi_train_10hr/chain/tdnn_lwf1a_fsh2cvi_sp//decode_graph_cvi_test_iterfinal/wer_8_0.0
# %WER 45.53 [ 17845 / 39195, 841 ins, 6574 del, 10430 sub ] exp_cvi_train_10hr/chain/tdnn_lwf1a_fsh2cvi_sp//decode_graph_fsh_dev_iterfinal/wer_7_0.0

# DenLWF (FSH --> CVI)
# %WER 51.12 [ 5051 / 9880, 354 ins, 1536 del, 3161 sub ] exp_cvi_train_10hr/chain/tdnn_denlwf1a_fsh2cvi_sp//decode_graph_cvi_test_iterfinal/wer_8_0.0
# %WER 40.16 [ 15741 / 39195, 930 ins, 5509 del, 9302 sub ] exp_cvi_train_10hr/chain/tdnn_denlwf1a_fsh2cvi_sp//decode_graph_fsh_dev_iterfinal/wer_7_0.0
