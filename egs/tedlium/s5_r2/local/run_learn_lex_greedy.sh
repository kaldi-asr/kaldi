#! /bin/bash
#
# This script demonstrates a lexicon learning recipe, which aims to imrove
# the pronounciation of abbreviated words in the TED-LIUM lexicon. It assumes
# the model exp/tri3 already exists. Please see steps/dict/learn_lexicon_greedy.sh
# for explanation of the options. 
#
# Copyright 2018  Xiaohui Zhang
# Apache 2.0

. ./cmd.sh
. ./path.sh

oov_symbol="<unk>"
# The user may have an phonetisaurus-trained English g2p model ready.
g2p_mdl_dir=
# The dir which contains the reference lexicon (most probably hand-derived)
# we want to expand/improve, and nonsilence_phones.txt,.etc which we need  
# for building new dict dirs.
ref_dict=data/local/dict
# acoustic training data we use to get alternative
# pronunciations and collet acoustic evidence.
data=data/train
# the cut-off parameter used to select pronunciation candidates from phone
# decoding. We remove pronunciations with probabilities less than this value
# after normalizing the probs s.t. the max-prob is 1.0 for each word."
min_prob=0.1
# Refer to steps/dict/select_prons_greedy.sh for the detailed meaning of
# alpha, beta and delta. Basically, the three dimensions of alpha
# and beta correspond to three pronunciation sources: phonetic-
# decoding, G2P and the reference lexicon, and the larger a value is,
# the more aggressive we'll prune pronunciations from that sooure.
# The valid range of each dim. is [0, 1] (for alpha, and 0 means 
# we never pruned pron from that source.) [0, 100] (for beta). 
alpha="0.04,0.02,0"
beta="30,5,0"
# Floor value of the pronunciation posterior statistics.
delta=0.00000001
# This parameter determines how many pronunciations we keep for each word
# after the first pass pruning. See steps/dict/internal/prune_pron_candidates.py
# for details.
vcr=16 

# Intermediate outputs of the lexicon learning stage will be put into dir
dir=exp/tri3_lex_greedy_work
nj=35
decode_nj=30
stage=0
lexlearn_stage=0
affix="learned_greedy"

. utils/parse_options.sh # accept options

# The reference vocab is the list of words which we already have hand-derived pronunciations.
ref_vocab=data/local/vocab.txt
cat $ref_dict/lexicon.txt | awk '{print $1}' | sort | uniq > $ref_vocab || exit 1; 

# Get a G2P generated lexicon for oov words (w.r.t the reference lexicon)
# in acoustic training data.
if [ $stage -le 0 ]; then
  if [ -z $g2p_mdl_dir ]; then
    g2p_mdl_dir=exp/g2p_phonetisaurus
    steps/dict/train_g2p_phonetisaurus.sh $ref_dict/lexicon.txt $g2p_mdl_dir || exit 1;
  fi
  awk '{for (n=2;n<=NF;n++) vocab[$n]=1;} END{for (w in vocab) printf "%s\n",w;}' \
    $data/text | sort -u > $data/train_vocab.txt || exit 1;
  awk 'NR==FNR{a[$1] = 1; next} {if(!($1 in a)) print $1}' $ref_vocab \
    $data/train_vocab.txt | sort > $data/oov_train.txt || exit 1;
  steps/dict/apply_g2p_phonetisaurus.sh --nbest 5 $data/train_vocab.txt $g2p_mdl_dir \
    exp/g2p_phonetisaurus/lex_train || exit 1;
fi

# Learn a lexicon based on the acoustic training data and the reference lexicon.
if [ $stage -le 1 ]; then
  steps/dict/learn_lexicon_greedy.sh --lexiconp-g2p "exp/g2p_phonetisaurus/lex_train/lexicon.lex" \
    --alpha $alpha --beta $beta --delta $delta \
    --min-prob $min_prob --cmd "$train_cmd" \
    --variant-counts-ratio $vcr \
    --stage $lexlearn_stage --nj 60 --oov-symbol $oov_symbol --retrain-src-mdl false \
    $ref_dict $ref_vocab $data exp/tri3 data/lang data/local/dict_${affix}_nosp \
    $dir || exit 1;
fi

# Add pronounciation probs to the learned lexicon.
if [ $stage -le 2 ]; then
  utils/prepare_lang.sh --phone-symbol-table data/lang/phones.txt \
    data/local/dict_${affix}_nosp $oov_symbol data/local/lang_${affix}_nosp data/lang_${affix}_nosp || exit 1;
  
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    $data data/lang_${affix}_nosp exp/tri2 exp/tri2_ali_${affix}_nosp || exit 1;
  
  steps/get_prons.sh --cmd "$train_cmd" data/train data/lang_${affix}_nosp exp/tri2_ali_${affix}_nosp || exit 1;
  
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_${affix}_nosp exp/tri2_ali_${affix}_nosp/pron_counts_nowb.txt \
    exp/tri2_ali_${affix}_nosp/sil_counts_nowb.txt \
    exp/tri2_ali_${affix}_nosp/pron_bigram_counts_nowb.txt data/local/dict_${affix} || exit 1;
  
  utils/prepare_lang.sh --phone-symbol-table data/lang/phones.txt \
    data/local/dict_${affix} $oov_symbol data/local/lang_${affix} data/lang_${affix} || exit 1;
fi

# Re-decode
if [ $stage -le 3 ]; then
  ! cmp data/lang_nosp/words.txt data/lang_${affix}/words.txt &&\
    echo "$0: The vocab of the affix lexicon and the reference vocab may be incompatible."
  cp data/lang_nosp/G.fst data/lang_${affix}/
  utils/mkgraph.sh data/lang_${affix} exp/tri3 exp/tri3/graph_${affix} || exit 1;
  
  for dset in dev test; do
  (  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri3/graph_${affix} data/${dset} exp/tri3/decode_${affix}_${dset} || exit 1;
  ) &
  done
fi

# RESULTS:
# Baseline:
# %WER 18.7 | 507 17783 | 83.9 11.4 4.7 2.6 18.7 92.3 | -0.006 | exp/tri3/decode_dev/score_17_0.0/ctm.filt.filt.sys
# %WER 17.6 | 1155 27500 | 84.7 11.6 3.7 2.4 17.6 87.2 | 0.013 | exp/tri3/decode_test/score_15_0.0/ctm.filt.filt.sys

# Re-decoding with the learned lexicon:
# %WER 18.5 | 507 17783 | 84.3 11.2 4.5 2.8 18.5 92.3 | -0.007 | exp/tri3/decode_learned_greedy_dev/score_16_0.0/ctm.filt.filt.sys
# %WER 17.5 | 1155 27500 | 84.9 11.5 3.6 2.4 17.5 87.5 | 0.035 | exp/tri3/decode_learned_greedy_test/score_14_0.0/ctm.filt.filt.sys

# To see the effect to neural-net results, one should re-train NN with the learned lexicon.
# Experiments have shown that, with the new lang dir, one should just re-run NN training
# starting from the supervision generation (steps/align_fmllr_lats.sh) stage, and should
# expect improved overall WERs and word recognition performance on words whose pronunciations
# were changed.

exit
wait
