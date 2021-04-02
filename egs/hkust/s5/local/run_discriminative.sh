#!/usr/bin/env bash
# Copyright 2016 Alibaba Robotics Corp. (Author: Xingyu Na)
# Apache 2.0

# This runs MMI and MPE on top of the MLE system. It requires the alignments.

dir=exp/tri5a

. ./cmd.sh
. ./path.sh

steps/make_denlats.sh --cmd "$train_cmd" --nj 10 --transform-dir ${dir}_ali \
  --config conf/decode.config \
  data/train data/lang $dir ${dir}_denlats || exit 1;

# Do MMI.
steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 \
  data/train data/lang ${dir}_ali ${dir}_denlats ${dir}_mmi_b0.1 || exit 1;

steps/decode.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
  --transform-dir $dir/decode \
  $dir/graph data/dev ${dir}_mmi_b0.1/decode || exit 1 ;

# Do MPE.
steps/train_mpe.sh  --cmd "$train_cmd" data/train data/lang ${dir}_ali ${dir}_denlats ${dir}_mpe || exit 1;

steps/decode.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
  --transform-dir $dir/decode \
  $dir/graph data/dev ${dir}_mpe/decode || exit 1 ;

exit 0;
