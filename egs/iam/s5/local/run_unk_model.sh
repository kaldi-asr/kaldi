#!/bin/bash


utils/lang/make_unk_lm.sh --ngram-order 3 --num-extra-ngrams 8300 data/train/dict exp/unk_lang_model

utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 \
  --unk-fst exp/unk_lang_model/unk_fst.txt data/train/dict "<unk>" data/lang/temp data/lang_unk

# note: it's important that the LM we built in data/lang/G.fst was created using
# pocolm with the option --limit-unk-history=true (see ted_train_lm.sh).  This
# keeps the graph compact after adding the unk model (we only have to add one
# copy of it).

cp data/lang_test_corpus/G.fst data/lang_unk/G.fst
#utils/mkgraph.sh data/lang_unk exp/tri3 exp/tri3/graph_unk

. ./cmd.sh

## Caution: if you use this unk-model stuff, be sure that the scoring script
## does not use lattice-align-words-lexicon, because it's not compatible with
## the unk-model.  Instead you should use lattice-align-words (of course, this
## only works if you have position-dependent phones).

#decode_=30
#for dset in dev test; do
#    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
#      exp/tri3/graph_unk data/${dset} exp/tri3/decode_${dset}_unk
#    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
#       data/${dset} exp/tri3/decode_${dset}_unk exp/tri3/decode_${dset}_unk_rescore
#done
#
#frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
#  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
#    --extra-left-context $chunk_left_context \
#    --extra-right-context $chunk_right_context \
#    --extra-left-context-initial 0 \
#    --extra-right-context-final 0 \
#    --frames-per-chunk $frames_per_chunk \
#    --nj $nj --cmd "$decode_cmd" \
#    $dir/graph data/test $dir/lang_test_corpus_c || exit 1
#
# # for x in exp/tri3/decode*; do grep Sum $x/*/*ys | utils/best_wer.sh ; done | grep -v old | grep -v si
