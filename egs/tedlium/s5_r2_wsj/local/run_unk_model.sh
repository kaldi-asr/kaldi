#!/bin/bash


utils/lang/make_unk_lm.sh data/local/dict exp/unk_lang_model

utils/prepare_lang.sh --unk-fst exp/unk_lang_model/unk_fst.txt data/local/dict "<unk>" data/local/lang data/lang_unk

cp data/lang/G.fst data/lang_unk/G.fst

utils/mkgraph.sh data/lang_unk exp/tri3 exp/tri3/graph_unk

. ./cmd.sh

## Caution: if you use this unk-model stuff, be sure that the scoring script
## does not use lattice-align-words-lexicon, because it's not compatible with
## the unk-model.  Instead you should use lattice-align-words (of course, this
## only works if you have position-dependent phones).

decode_nj=30
for dset in dev test; do
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri3/graph_unk data/${dset} exp/tri3/decode_${dset}_unk
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
       data/${dset} exp/tri3/decode_${dset}_unk exp/tri3/decode_${dset}_unk_rescore
done

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
