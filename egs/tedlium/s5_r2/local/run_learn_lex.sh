#! /bin/bash
#
# This script demonstrates a lexicon learning recipe, which aims to imrove
# the pronounciation of abbreviated words in the TED-LIUM lexicon. It assumes
# the model exp/tri3 already exists. Please see steps/dict/learn_lexicon.sh
# for explanation of the options. 
#
# Copyright 2016  Xiaohui Zhang
# Apache 2.0

. cmd.sh
. path.sh

oov_symbol="<unk>"
ref_dict=data/local/dict
# The dir which contains the reference lexicon (most probably hand-derived)"
# we want to expand/improve, and nonsilence_phones.txt,.etc which we need " 
# for building new dict dirs."
data=data/train
# acoustic training data we use to get alternative"
# pronunciations and collet acoustic evidence."
min_prob=0.4
# the cut-off parameter used to select pronunciation candidates from phone
# decoding. A smaller min-prob means more candidates will be included.
prior_mean="0.7,0.2,0.1"        
# Mean of priors (summing up to 1) assigned to three exclusive pronunciation"
# source: reference lexicon, g2p, and phone decoding (used in the Bayesian"
# pronunciation selection procedure). We recommend setting a larger prior"
# mean for the reference lexicon, e.g. '0.6,0.2,0.2'."
prior_counts_tot=15
# Total amount of prior counts we add to all pronunciation candidates of"
# each word. By timing it with the prior mean of a source, and then dividing"
# by the number of candidates (for a word) from this source, we get the"
# prior counts we actually add to each candidate."
variants_prob_mass=0.6
# In the Bayesian pronunciation selection procedure, for each word, we"
# choose candidates (from all three sources) with highest posteriors"
# until the total prob mass hit this amount."
# It's used in a similar fashion when we apply G2P."
variants_prob_mass2=0.9
# In the Bayesian pronunciation selection procedure, for each word,"
# after the total prob mass of selected candidates hit variants-prob-mass,"
# we continue to pick up reference candidates with highest posteriors"
# until the total prob mass hit this amount."
affix="lex"
# Intermediate outputs of the lexicon learning stage will be put into
# exp/tri3_${affix}_work
nj=35
decode_nj=30
lexlearn_stage=0
stage=0


. utils/parse_options.sh # accept options

# learn a lexicon based on the acoustic training data and the reference lexicon.
if [ $stage -le 0 ]; then
  ref_vocab=data/local/vocab # The reference vocab is the vocab we aim to generate pronounciations for.
  cat $ref_dict/lexicon.txt | cut -f1 -d' ' | sort | uniq > $ref_vocab
  steps/dict/learn_lexicon.sh --min-prob $min_prob --variants-prob-mass $variants_prob_mass --variants-prob-mass2 $variants_prob_mass2  \
    --prior-counts-tot $prior_counts_tot --prior-mean $prior_mean \
    --affix $affix --stage $lexlearn_stage \
    --nj 60 --oov-symbol $oov_symbol --retrain-src-mdl false --g2p-for-iv false \
    $ref_dict $ref_vocab $data exp/tri3 data/lang data/local/dict_learned_nosp
fi

# add pronounciation probs to the learned lexicon.
if [ $stage -le 1 ]; then
  utils/prepare_lang.sh --phone-symbol-table data/lang/phones.txt \
    data/local/dict_learned_nosp $oov_symbol data/local/lang_learned_nosp data/lang_learned_nosp
  
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    $data data/lang_learned_nosp exp/tri2 exp/tri2_ali_learned_lex_nosp
  
  steps/get_prons.sh --cmd "$train_cmd" data/train data/lang_learned_nosp exp/tri2_ali_learned_lex_nosp
  
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_learned_nosp exp/tri2_ali_learned_lex_nosp/pron_counts_nowb.txt \
    exp/tri2_ali_learned_lex_nosp/sil_counts_nowb.txt \
    exp/tri2_ali_learned_lex_nosp/pron_bigram_counts_nowb.txt data/local/dict_learned
  
  utils/prepare_lang.sh --phone-symbol-table data/lang/phones.txt \
    data/local/dict_learned $oov_symbol data/local/lang_learned data/lang_learned
fi

# re-train the acoustic model using the learned lexicon
if [ $stage -le 2 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $data data/lang_learned exp/tri3 exp/tri3_ali_learned_lex
  
  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 $data data/lang_learned exp/tri3_ali_learned_lex exp/tri3_learned_lex
fi

# decode
if [ $stage -le 3 ]; then
  cp -rT data/lang_learned data/lang_learned_rescore
  ! cmp data/lang_nosp/words.txt data/lang_learned/words.txt &&\
    echo "$0: The vocab of the learned lexicon and the reference vocab may be incompatible."
  cp data/lang_nosp/G.fst data/lang_learned/
  cp data/lang_nosp_rescore/G.carpa data/lang_learned_rescore/
  utils/mkgraph.sh data/lang_learned exp/tri3_learned_lex exp/tri3_learned_lex/graph
  
  for dset in dev test; do
  (  steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri3_learned_lex/graph data/${dset} exp/tri3_learned_lex/decode_${dset}
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_learned data/lang_learned_rescore \
       data/${dset} exp/tri3_learned_lex/decode_${dset} exp/tri3_learned_lex/decode_${dset}_rescore
  ) &
  done
fi
