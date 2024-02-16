#!/bin/bash

# Copyright 2013 Idiap Research Institute (Author: David Imseng)
# Apache 2.0

. cmd.sh

big_memory_cmd="$decode_cmd --mem 8G"

states=20000
dir=exp/tri4b_pretrain-dbn_dnn/

steps/kl_hmm/build_tree.sh --cmd "$big_memory_cmd" --thresh -1 --nnet_dir exp/tri4b_pretrain-dbn_dnn/ \
 ${states} data-fmllr-tri4b/train_si284 data/lang exp/tri4b_ali_si284 exp/tri4b-${states} || exit 1;

utils/mkgraph.sh data/lang_test_bd_tgpr exp/tri4b-${states} exp/tri4b-${states}/graph_bd_tgpr || exit 1;

steps/kl_hmm/train_kl_hmm.sh --nj 30 --cmd "$big_memory_cmd" --model exp/tri4b-${states}/final.mdl data-fmllr-tri4b/train_si284 exp/tri4b-${states} $dir/kl-hmm-${states}

steps/kl_hmm/decode_kl_hmm.sh --nj 10 --cmd "$big_memory_cmd" --acwt 0.1 --nnet $dir/kl-hmm-${states}/final.nnet --model exp/tri4b-${states}/final.mdl \
  --config conf/decode_dnn.config exp/tri4b-${states}/graph_bd_tgpr/ data-fmllr-tri4b/test_dev93 $dir/decode_dev93_kl-hmm-bd-${states}_tst

steps/kl_hmm/decode_kl_hmm.sh --nj 8 --cmd "$big_memory_cmd" --acwt 0.1 --nnet $dir/kl-hmm-${states}/final.nnet --model exp/tri4b-${states}/final.mdl \
  --config conf/decode_dnn.config exp/tri4b-${states}/graph_bd_tgpr/ data-fmllr-tri4b/test_eval92 $dir/decode_eval92_kl-hmm-bd-${states}_tst


