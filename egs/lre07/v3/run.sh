#!/bin/bash
# Copyright  2014   David Snyder
# Apache 2.0.

# This script is an example of a phonotactic system for language
# identification on the NIST LRE 2007 closet-set evaluation.

. cmd.sh
. path.sh
set -e

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
languages=local/general_lr_closed_set_langs.txt

if [ 0 = 1 ]; then
# Training data sources
local/make_sre_2008_train.pl /export/corpora5/LDC/LDC2011S05 data
local/make_callfriend.pl /export/corpora/LDC/LDC96S60 vietnamese data
local/make_callfriend.pl /export/corpora/LDC/LDC96S59 tamil data
local/make_callfriend.pl /export/corpora/LDC/LDC96S53 japanese data
local/make_callfriend.pl /export/corpora/LDC/LDC96S52 hindi data
local/make_callfriend.pl /export/corpora/LDC/LDC96S51 german data
local/make_callfriend.pl /export/corpora/LDC/LDC96S50 farsi data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S48 french data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S49 arabic.standard data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S54 korean data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S55 chinese.mandarin.mainland data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S56 chinese.mandarin.taiwan data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S57 spanish.caribbean data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S58 spanish.noncaribbean data
local/make_lre96.pl /export/corpora/NIST/lid96e1 data
local/make_lre03.pl /export/corpora4/LDC/LDC2006S31 data
local/make_lre05.pl /export/corpora5/LDC/LDC2008S05 data
local/make_lre07_train.pl /export/corpora5/LDC/LDC2009S05 data
local/make_lre09.pl /export/corpora5/NIST/LRE/LRE2009/eval data

# Make the evaluation data set. We're concentrating on the General Language
# Recognition Closet-Set evaluation, so we remove the dialects and filter
# out the unknown languages used in the open-set evaluation.
local/make_lre07.pl /export/corpora5/LDC/LDC2009S04 data/lre07_all

cp -r data/lre07_all data/lre07
utils/filter_scp.pl -f 2 $languages <(lid/remove_dialect.pl data/lre07_all/utt2lang) \
  > data/lre07/utt2lang
utils/fix_data_dir.sh data/lre07

src_list="data/sre08_train_10sec_female \
    data/sre08_train_10sec_male data/sre08_train_3conv_female \
    data/sre08_train_3conv_male data/sre08_train_8conv_female \
    data/sre08_train_8conv_male data/sre08_train_short2_male \
    data/sre08_train_short2_female data/ldc96* data/lid05d1 \
    data/lid05e1 data/lid96d1 data/lid96e1 data/lre03 \
    data/ldc2009* data/lre09"

# Remove any spk2gender files that we have: since not all data
# sources have this info, it will cause problems with combine_data.sh
for d in $src_list; do rm -f $d/spk2gender 2>/dev/null; done

utils/combine_data.sh data/train_unsplit_all $src_list
fi

utils/apply_map.pl -f 2 --permissive local/lang_map.txt \
  < data/train_unsplit/utt2lang  2>/dev/null > foo

cp foo data/train_unsplit_all/utt2lang
cp -r data/train_unsplit_all data/train_unsplit

lid/remove_dialect.pl data/train_unsplit_all/utt2lang > foo
utils/filter_scp.pl -f 2 $languages foo \
  > data/train_unsplit/utt2lang
utils/fix_data_dir.sh data/train_unsplit
rm foo

echo "**Language count in training:**"
awk '{print $2}' data/train_unsplit/utt2lang | sort | uniq -c | sort -nr


local/split_long_utts.sh --max-utt-len 30 data/train_unsplit data/train


# This commented script is an alternative to the above utterance
# splitting method. Here we split the utterance based on the number of 
# frames which are voiced, rather than the total number of frames.
# max_voiced=3000 
# local/vad_split_utts.sh --max-voiced $max_voiced data/train_unsplit $mfccdir data/train

steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 100 --cmd "$train_cmd" \
  data/train exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/lre07 exp/make_mfcc $mfccdir

lid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/train \
  exp/make_vad $vaddir
lid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/lre07 \
  exp/make_vad $vaddir

steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir 
steps/compute_cmvn_stats.sh data/lre07 exp/make_mfcc/lre07 $mfccdir

# Prepare data for phone recognizer training.
local/fisher_data_prep.sh /export/corpora3/LDC/LDC2004T19 /export/corpora3/LDC/LDC2005T19 \
   /export/corpora3/LDC/LDC2004S13 /export/corpora3/LDC/LDC2005S13

local/fisher_prepare_dict.sh

utils/prepare_lang.sh --position-dependent-phones false data/local/dict "<unk>" data/local/lang data/lang

utils/fix_data_dir.sh data/train_phonotactics

steps/make_mfcc.sh --nj 40 --cmd "$train_cmd" data/train_phonotactics exp/make_mfcc/train_phonotactics $mfccdir || exit 1;
utils/fix_data_dir.sh data/train_phonotactics
steps/compute_cmvn_stats.sh data/train_phonotactics exp/make_mfcc/train_phonotactics $mfccdir 
utils/subset_data_dir.sh data/train_phonotactics 500000 data/train_phonotactics_500k

utils/fix_data_dir.sh data/train_phonotactics_500k
utils/validate_data_dir.sh data/train_phonotactics_500k

utils/subset_data_dir.sh --shortest data/train_phonotactics_500k 100000 data/train_phonotactics_100kshort
utils/subset_data_dir.sh data/train_phonotactics_100kshort 10000 data/train_phonotactics_10k
local/remove_dup_utts.sh 100 data/train_phonotactics_10k data/train_phonotactics_10k_nodup
utils/subset_data_dir.sh --speakers data/train_phonotactics_500k 30000 data/train_phonotactics_30k
utils/subset_data_dir.sh --speakers data/train_phonotactics_500k 100000 data/train_phonotactics_100k

steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
  data/train_phonotactics_10k_nodup data/lang exp/mono0a 

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_phonotactics_30k data/lang exp/mono0a exp/mono0a_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 data/train_phonotactics_30k data/lang exp/mono0a_ali exp/tri1 || exit 1;

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_phonotactics_30k data/lang exp/tri1 exp/tri1_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 data/train_phonotactics_30k data/lang exp/tri1_ali exp/tri2 || exit 1;

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_phonotactics_500k data/lang exp/tri2 exp/tri2_ali || exit 1;

# Train tri3a, which is LDA+MLLT, on 500k data.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   5000 40000 data/train_phonotactics_500k data/lang exp/tri2_ali exp/tri3a || exit 1;

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_phonotactics_30k data/lang exp/tri3a exp/tri3a_ali || exit 1;

steps/train_sat.sh  --cmd "$train_cmd" \
  2200 25000 data/train_phonotactics_30k data/lang exp/tri3a_ali exp/tri4a || exit 1;

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_phonotactics_30k data/lang exp/tri4a exp/tri4a_ali || exit 1;

# Reduce the number of gaussians
steps/train_sat.sh  --cmd "$train_cmd" \
  2600 30000 data/train_phonotactics_30k data/lang exp/tri4a_ali exp/tri5a || exit 1;

# Create the grammar FST
ali-to-phones exp/tri5a/final.mdl "ark:gunzip -c exp/tri5a/ali.*.gz|" ark,t:- | \
 awk '{for (n=2;n<=NF;n++) { printf("%s ", $n); } printf("\n"); }'  | \
 local/make_phone_bigram.pl | fstcompile  > data/lang/G.fst

cp data/lang/phones.txt data/lang/words.txt # same symbol table for words and phones.

# Now make lexicon: one word per phone.  No optional silence.
cat data/lang/phones.txt | grep -v '#' | grep -v -w '<eps>' | \
   awk '{printf("0 0 %s %s 0\n", $2, $2); } END{ printf("0 0\n"); }' | \
  fstcompile > data/lang/L.fst

cp data/lang/L.fst data/lang/L_disambig.fst

rm -f data/lang/phones/word_boundary.txt 2>/dev/null 

utils/mkgraph.sh data/lang exp/tri5a exp/tri5a/graph || exit 1;

steps/get_fmllr_basis.sh --cmd "$train_cmd" data/train_phonotactics_100k data/lang exp/tri5a

local/decode_basis_fmllr.sh --nj 25 --acwt 0.075 --num-threads 8 --parallel-opts "-pe smp 8" --cmd "$decode_cmd" \
   --skip-scoring true --config conf/decode.config \
   exp/tri5a/graph data/lre07 exp/tri5a/lre07_basis_fmllr

local/decode_basis_fmllr.sh  --nj 50 --acwt 0.075 --num-threads 8 --parallel-opts "-pe smp 8" --cmd "$decode_cmd" \
   --skip-scoring true --config conf/decode.config \
   exp/tri5a/graph data/train exp/tri5a/train_basis_fmllr

local/make_softcount_feats.sh

local/run_logistic_regression_phonotactics.sh

# General LR 2007 closed-set eval
local/lre07_eval/lre07_eval.sh exp/ivectors_lre07 \
  local/general_lr_closed_set_langs.txt
