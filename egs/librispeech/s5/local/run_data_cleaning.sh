#!/usr/bin/env bash


# This script shows how you can do data-cleaning, and exclude data that has a
# higher likelihood of being wrongly transcribed. This didn't help with the 
# LibriSpeech data set, so perhaps the data is already clean enough.
# For the actual results see the comments at the bottom of this script.

stage=1
. ./cmd.sh || exit 1;


. utils/parse_options.sh || exit 1;

set -e


if [ $stage -le 1 ]; then
  steps/cleanup/find_bad_utts.sh --nj 100 --cmd "$train_cmd" data/train_960 data/lang \
    exp/tri6b exp/tri6b_cleanup
fi

thresh=0.1
if [ $stage -le 2 ]; then
  cat exp/tri6b_cleanup/all_info.txt | awk -v threshold=$thresh '{ errs=$2;ref=$3; if (errs <= threshold*ref) { print $1; } }' > uttlist
  utils/subset_data_dir.sh --utt-list uttlist data/train_960 data/train_960_thresh$thresh
fi

if [ $stage -le 3 ]; then
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train.thresh$thresh data/lang exp/tri6b exp/tri6b_ali_$thresh
fi

if [ $stage -le 4 ]; then
  steps/train_sat.sh  --cmd "$train_cmd" \
    7000 150000 data/train_960_thresh$thresh data/lang exp/tri6b_ali_$thresh  exp/tri6b_$thresh || exit 1;
fi

if [ $stage -le 5 ]; then
  utils/mkgraph.sh data/lang_test_tgsmall exp/tri6b_$thresh exp/tri6b_$thresh/graph_tgsmall || exit 1
  for test in dev_clean dev_other; do
    steps/decode_fmllr.sh --nj 50 --cmd "$decode_cmd" --config conf/decode.config \
      exp/tri6b_$thresh/graph_tgsmall data/$test exp/tri6b_$thresh/decode_tgsmall_$test || exit 1
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test exp/tri6b_$thresh/decode_{tgsmall,tgmed}_$test  || exit 1;
  done
fi


# # Results with the original data
# %WER 9.37 [ 5095 / 54402, 607 ins, 646 del, 3842 sub ] exp/tri6b/decode_tgmed_dev_clean/wer_14
# %WER 24.83 [ 12650 / 50948, 1136 ins, 2029 del, 9485 sub ] exp/tri6b/decode_tgmed_dev_other/wer_16
# %WER 10.77 [ 5857 / 54402, 662 ins, 769 del, 4426 sub ] exp/tri6b/decode_tgsmall_dev_clean/wer_13
# %WER 27.05 [ 13781 / 50948, 1193 ins, 2306 del, 10282 sub ] exp/tri6b/decode_tgsmall_dev_other/wer_15

# # Results with cleaned up subset of the data (at threshold 0.1)
# %WER 9.37 [ 5099 / 54402, 611 ins, 634 del, 3854 sub ] exp/tri6b_0.1/decode_tgmed_dev_clean/wer_14
# %WER 24.93 [ 12699 / 50948, 1166 ins, 1959 del, 9574 sub ] exp/tri6b_0.1/decode_tgmed_dev_other/wer_16
# %WER 10.66 [ 5799 / 54402, 665 ins, 757 del, 4377 sub ] exp/tri6b_0.1/decode_tgsmall_dev_clean/wer_13
# %WER 27.07 [ 13790 / 50948, 1130 ins, 2348 del, 10312 sub ] exp/tri6b_0.1/decode_tgsmall_dev_other/wer_16


