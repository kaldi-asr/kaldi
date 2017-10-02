#!/bin/bash


utils/lang/make_unk_lm.sh --ngram-order 3 --num-extra-ngrams 8300 data/train/dict exp/unk_lang_model

utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 \
  --unk-fst exp/unk_lang_model/unk_fst.txt data/train/dict "<unk>" data/lang/temp data/lang_unk

# note: it's important that the LM we built in data/lang/G.fst was created using
# pocolm with the option --limit-unk-history=true (see ted_train_lm.sh).  This
# keeps the graph compact after adding the unk model (we only have to add one
# copy of it).

cp data/lang_test_corpus/G.fst data/lang_unk/G.fst


utils/mkgraph.sh --self-loop-scale 1.0 data/lang_unk exp/mono exp/mono/graph_unk_2 || exit 1;
#utils/mkgraph.sh --self-loop-scale 1.0 data/lang_unk exp/mono_8states_4sil_10000 exp/mono_8states_4sil_10000/graph_unk_1 || exit 1;
#utils/mkgraph.sh --self-loop-scale 1.0 data/lang_unk exp/tri3_8states_4sil_10000_500_20000_500_20000_500_20000 exp/tri3_8states_4sil_10000_500_20000_500_20000_500_20000/graph_unk || exit 1;
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

# # dev results.  unk-model helps slightly before rescoring.
# %WER 19.3 | 507 17783 | 83.7 11.6 4.7 3.0 19.3 91.5 | -0.076 | exp/tri3/decode_dev/score_17_0.0/ctm.filt.filt.sys
# %WER 18.2 | 507 17783 | 84.8 10.7 4.5 3.0 18.2 91.3 | -0.111 | exp/tri3/decode_dev_rescore/score_16_0.0/ctm.filt.filt.sys
# %WER 19.1 | 507 17783 | 83.7 11.3 5.1 2.8 19.1 91.9 | -0.044 | exp/tri3/decode_dev_unk/score_17_0.0/ctm.filt.filt.sys
# %WER 18.2 | 507 17783 | 84.5 10.6 4.9 2.8 18.2 91.5 | -0.047 | exp/tri3/decode_dev_unk_rescore/score_15_0.0/ctm.filt.filt.sys


# # dev results.  unk-model helps slightly after rescoring.
# %WER 17.3 | 1155 27500 | 85.0 11.5 3.5 2.4 17.3 86.9 | -0.035 | exp/tri3/decode_test/score_15_0.0/ctm.filt.filt.sys
# %WER 16.6 | 1155 27500 | 85.8 11.0 3.2 2.4 16.6 86.4 | -0.098 | exp/tri3/decode_test_rescore/score_14_0.0/ctm.filt.filt.sys
# %WER 17.3 | 1155 27500 | 84.9 11.3 3.8 2.2 17.3 87.4 | -0.015 | exp/tri3/decode_test_unk/score_15_0.0/ctm.filt.filt.sys
# %WER 16.5 | 1155 27500 | 85.7 10.7 3.6 2.2 16.5 86.7 | -0.075 | exp/tri3/decode_test_unk_rescore/score_14_0.0/ctm.filt.filt.sys
