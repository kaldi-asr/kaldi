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
prior_counts="2.5-1-1"
min_prob=0.3
var_mass=0.7
ref_dict=data/local/dict
data=data/train
affix="lex"
nj=35
decode_nj=30
lexlearn_stage=1
stage=0

. utils/parse_options.sh # accept options

# learn a new lexicon
if [ $stage -le 0 ]; then
  ref_vocab=data/local/vocab # The reference vocab is the vocab we aim to generate pronounciations for.
  cat $ref_dict/lexicon.txt | cut -f1 -d' ' | sort | uniq > $ref_vocab
  steps/dict/learn_lexicon.sh --min-prob $min_prob --var-mass $var_mass \
    --prior-counts $prior_counts --affix $affix --stage $lexlearn_stage \
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
fi

# re-train the acoustic model using the learned lexicon
if [ $stage -le 2 ]; then
  utils/prepare_lang.sh --phone-symbol-table data/lang/phones.txt \
    data/local/dict_learned $oov_symbol data/local/lang_learned data/lang_learned
  
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    $data data/lang_learned exp/tri3 exp/tri3_ali_learned_lex
  
  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 $data data/lang_learned exp/tri3_ali_learned_lex exp/tri3_learned_lex
fi

# decode
if [ $stage -le 3 ]; then
  cp -rT data/lang_learned data/lang_learned_rescore
  cp data/lang_nosp/G.fst data/lang_learned/
  cp data/lang_nosp_rescore/G.carpa data/lang_learned_rescore/
  utils/mkgraph.sh data/lang_learned exp/tri3_learned_lex exp/tri3_learned_lex/graph
  
  for dset in dev test; do
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri3_learned_lex/graph data/${dset} exp/tri3_learned_lex/decode_${dset}
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_learned data/lang_learned_rescore \
       data/${dset} exp/tri3_learned_lex/decode_${dset} exp/tri3_learned_lex/decode_${dset}_rescore
  done
fi
