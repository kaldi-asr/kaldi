#!/bin/bash
#
# Based mostly on the Switchboard recipe. The training database is TED-LIUM,
# it consists of TED talks with cleaned automatic transcripts:
#
# http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus
# http://www.openslr.org/resources (Mirror).
#
# The data is distributed under 'Creative Commons BY-NC-ND 3.0' license,
# which allow free non-commercial use, while only a citation is required.
#
# Copyright  2014 Nickolay V. Shmyrev 
#            2014 Brno University of Technology (Author: Karel Vesely)
# Apache 2.0
#

# TODO : use pruned trigram?

. cmd.sh
. path.sh

nj=80
decode_nj=8

stage=0
. utils/parse_options.sh # accept options

# Data preparation
if [ $stage -le 0 ]; then
  local/download_data.sh || exit 1

  local/prepare_data.sh || exit 1

  local/prepare_dict.sh || exit 1

  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang || exit 1

  local/prepare_lm.sh || exit 1
fi

# Feature extraction
feat_dir=$pwd/data/mfcc_features
if [ $stage -le 1 ]; then
  for set in test dev train; do
    dir=data/$set
    steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" $dir $dir/log $dir/data || exit 1
    steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1
  done
fi

# Now we have 118 hours of training data.
# Let's create a subset with 10k short segments to make flat-start training easier:
if [ $stage -le 2 ]; then
  utils/subset_data_dir.sh --shortest data/train 10000 data/train_10kshort || exit 1
  local/remove_dup_utts.sh 10 data/train_10kshort data/train_10kshort_nodup || exit 1
fi

# Train
if [ $stage -le 3 ]; then
  steps/train_mono.sh --nj 20 --cmd "$train_cmd" \
    data/train_10kshort_nodup data/lang exp/mono0a || exit 1

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/mono0a exp/mono0a_ali || exit 1

  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 30000 data/train data/lang exp/mono0a_ali exp/tri1 || exit 1

  utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1

  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
    --num-threads 4 --parallel-opts "-pe smp 4" \
    exp/tri1/graph data/dev exp/tri1/decode_dev || exit 1
  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
    --num-threads 4 --parallel-opts "-pe smp 4" \
    exp/tri1/graph data/test exp/tri1/decode_test || exit 1
fi

if [ $stage -le 4 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    4000 50000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1

  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph || exit 1

  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
    --num-threads 4 --parallel-opts "-pe smp 4" \
    exp/tri2/graph data/dev exp/tri2/decode_dev || exit 1
  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
    --num-threads 4 --parallel-opts "-pe smp 4" \
    exp/tri2/graph data/test exp/tri2/decode_test || exit 1
fi

if [ $stage -le 5 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali || exit 1

  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 data/train data/lang exp/tri2_ali exp/tri3 || exit 1

  utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph || exit 1

  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" \
    --num-threads 4 --parallel-opts "-pe smp 4" \
    exp/tri3/graph data/dev exp/tri3/decode_dev || exit 1
  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" \
    --num-threads 4 --parallel-opts "-pe smp 4" \
    exp/tri3/graph data/test exp/tri3/decode_test || exit 1
fi

if [ $stage -le 6 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri3 exp/tri3_ali || exit 1

  steps/make_denlats.sh --transform-dir exp/tri3_ali --nj $nj --cmd "$decode_cmd" \
    data/train data/lang exp/tri3 exp/tri3_denlats || exit 1

  steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 \
    data/train data/lang exp/tri3_ali exp/tri3_denlats \
    exp/tri3_mmi_b0.1 || exit 1

  for iter in 4; do
  steps/decode.sh --transform-dir exp/tri3/decode_dev --nj $decode_nj --cmd "$decode_cmd" --iter $iter \
    --num-threads 4 --parallel-opts "-pe smp 4" \
    exp/tri3/graph data/dev exp/tri3_mmi_b0.1/decode_dev_it$iter || exit 1
  steps/decode.sh --transform-dir exp/tri3/decode_test --nj $decode_nj --cmd "$decode_cmd" --iter $iter \
    --num-threads 4 --parallel-opts "-pe smp 4" \
    exp/tri3/graph data/test exp/tri3_mmi_b0.1/decode_test_it$iter || exit 1
  done
fi

# Run the DNN recipe on fMLLR feats:
local/run_dnn.sh || exit 1

echo success...
exit 0
