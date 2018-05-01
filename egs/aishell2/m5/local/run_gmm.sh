#!/bin/bash
# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
# Apache 2.0

# number of jobs
nj=20

. ./cmd.sh
[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh

# Now make MFCC plus pitch features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc
for x in train dev test; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  utils/fix_data_dir.sh data/$x || exit 1;
done

# subset the training data for fast startup
for x in 100 300; do
	utils/subset_data_dir.sh data/train ${x}000 data/train_${x}k
done

# mono training
steps/train_mono.sh --cmd "$train_cmd" --nj $nj \
  data/train_100k data/lang exp/mono || exit 1;

# mono decoding
utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj $nj \
  exp/mono/graph data/dev exp/mono/decode_dev
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj $nj \
  exp/mono/graph data/test exp/mono/decode_test

# mono alignment
steps/align_si.sh --cmd "$train_cmd" --nj $nj \
  data/train_300k data/lang exp/mono exp/mono_ali || exit 1;

# tri1 training
steps/train_deltas.sh --cmd "$train_cmd" \
 4000 32000 data/train_300k data/lang exp/mono_ali exp/tri1 || exit 1;

# tri1 decoding
utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${nj} \
  exp/tri1/graph data/dev exp/tri1/decode_dev
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${nj} \
  exp/tri1/graph data/test exp/tri1/decode_test

# tri1 alignment
steps/align_si.sh --cmd "$train_cmd" --nj $nj \
  data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

# tri2 training
steps/train_deltas.sh --cmd "$train_cmd" \
 7000 56000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1;

# tri2 decoding
utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${nj} \
  exp/tri2/graph data/dev exp/tri2/decode_dev
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${nj} \
  exp/tri2/graph data/test exp/tri2/decode_test

# tri2 alignment
steps/align_si.sh --cmd "$train_cmd" --nj $nj \
  data/train data/lang exp/tri2 exp/tri2_ali || exit 1;

# tri3 training [LDA+MLLT]
steps/train_lda_mllt.sh --cmd "$train_cmd" \
 10000 80000 data/train data/lang exp/tri2_ali exp/tri3 || exit 1;

# tri3 decoding
utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph || exit 1;
steps/decode.sh --cmd "$decode_cmd" --nj ${nj} --config conf/decode.conf \
  exp/tri3/graph data/dev exp/tri3/decode_dev
steps/decode.sh --cmd "$decode_cmd" --nj ${nj} --config conf/decode.conf \
  exp/tri3/graph data/test exp/tri3/decode_test

# tri3 alignment
steps/align_si.sh --cmd "$train_cmd" --nj $nj \
  data/train data/lang exp/tri3 exp/tri3_ali || exit 1;

steps/align_si.sh --cmd "$train_cmd" --nj ${nj} \
  data/dev data/lang exp/tri3 exp/tri3_ali_dev || exit 1;

echo "local/run_gmm.sh succeeded"
exit 0;
