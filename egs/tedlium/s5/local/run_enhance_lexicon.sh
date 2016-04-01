#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0

. cmd.sh
. path.sh

dir=exp/tri3_debug_lexicon
mkdir -p $dir

############################################################################### 
## Train G2P model on the original lexicon
############################################################################### 
steps/dict/train_g2p.sh --cmd "$decode_cmd --mem 4G" \
  data/local/dict/lexicon.txt exp/g2p

############################################################################### 
## Get list of train words and oovs
############################################################################### 
cat data/train/text | python -c 'import sys
words = {}
for line in open(sys.argv[1]):
  line.strip.splits()
  words[splits[0]] = 1

train_words = {}
for line in sys.stdin.readlines():
  line.strip.splits()
  for x in splits[1:]:
    train_words[x] = 1

train_words_handle = open(sys.argv[2], "w")
oov_words_handle = open(sys.argv[3], "w")

for x in train_words:
  train_words_handle.write(x + "\n")
  if x not in words:
    oov_words_handle.write(x + "\n")
    ' data/lang/words.txt $dir/train_words.txt.unsorted $dir/oovs.txt.unsorted

sort $dir/oovs.txt.unsorted > $dir/oovs.txt
sort $dir/train_words.txt.unsorted > $dir/train_words.txt

############################################################################### 
## Apply G2P model on the oovs
############################################################################### 
mkdir -p $dir/oov_lex5
steps/dict/apply_g2p.sh --var-counts 5 exp/tri3_debug_lexicon/oovs.txt exp/g2p $dir/oov_lex5

############################################################################### 
# Prepare dict and lang directories by combining the lexicon for 
# OOVs that is obtained from G2P
############################################################################### 

# Add NSN as pronunciation to words that G2P is not able to get pronunciations for
utils/filter_scp.pl --exclude $dir/oov_lex5/lexicon.lex $dir/oov_lex5/wordlist.txt | \
  awk '{print $1"\t1.0\tNSN"}' > $dir/oov_lex5/oov.lex

# Create extended dict directory, but ignoring probabilities
cp -r data/local/dict_nosp/ data/local/dict_extended_nosp
cat data/local/dict_nosp/lexiconp.txt $dir/oov_lex5/lexicon.lex \
  $dir/oov_lex5/oov.lex | \
  awk '{printf $1; for (i=3; i<=NF; i++) printf "\t"$1;}' > \
  data/local/dict_extended_nosp/lexicon.txt 
rm data/local/dict_extended_nosp/lexiconp.txt 
rm data/local/dict_extended_nosp/lexicon_words.txt 

# Prepare lang directory
utils/prepare_lang.sh data/local/dict_extended_nosp/ "<unk>" data/{local/,}lang_extended_nosp 

############################################################################### 
# Run debug_lexicon script using the extended lexicon
############################################################################### 
extended_debug_dir=exp/tri3_debug_extended_lexicon
steps/dict/debug_lexicon.sh --nj 100 --cmd queue.pl data/train data/lang_extended_nosp/ \
  exp/tri3 data/local/dict_extended_nosp/lexicon.txt $dir

# Add more candidates to the extended lexicon
python steps/dict/prons_to_lexicon.py < \
  $extended_debug_dir/prons.txt > \
  $extended_debug_dir/debug_extended_lexicon.lex

cp -r data/local/dict_nosp/ data/local/dict_tri3b_nosp
rm data/local/dict_tri3b_nosp/lexiconp.txt 
rm data/local/dict_tri3b_nosp/lexicon_words.txt 

############################################################################### 
# Create tri3b lexicon and lang
############################################################################### 
cat $extended_debug_dir/debug_extended_lexiconp.lex | \
  grep -vP "<eps>|\[.*\]" | awk '{if (NF > 2) print $0}' > \
  $extended_debug_dir/debug_extended_filt_lexiconp.lex
cat $debug_dir/oov_lex5/lexicon.lex $extended_debug_dir/debug_extended_filt_lexiconp.lex \
  <(utils/apply_map.pl -f 3- $extended_debug_dir/phone_map.txt < data/local/lang/lexiconp.txt) | \
  tr ' ' '\t' | awk '{if (NF >=3) print $0}' |cut -f 1,3- | sort -u > data/local/dict_tri3b_nosp/lexicon.txt
rm data/local/dict_tri3b_nosp/lexiconp.txt
utils/prepare_lang.sh data/local/dict_tri3b_nosp/ "<unk>" data/{local/,}lang_tri3b_nosp

############################################################################### 
# Create tri3b lexicon and lang
############################################################################### 
lats_dir=exp/tri3_tri3b_nosp_lex_lats

steps/align_fmllr_lats.sh --cmd "$train_cmd" --nj 100 \
  data/train data/lang_tri3b_nosp exp/tri3 $lats_dir

# Get arc level information from the lattice
$train_cmd JOB=1:100 $lats_dir/arc_info/log/get_arc_info.JOB.log \
  lattice-align-words data/lang_tri3b_nosp/phones/word_boundary.int \
  $lats_dir/final.mdl \
  "ark:gunzip -c $lats_dir/lat.JOB.gz |" ark:- \| \
  lattice-arc-post --acoustic-scale=0.1 $lats_dir/final.mdl ark:- - \| \
  utils/int2sym.pl -f 5 data/lang_tri3b_nosp/words.txt \| \
  utils/int2sym.pl -f 6- data/lang_tri3b_nosp/phones.txt '>' \
  $lats_dir/arc_info/arc_info_sym.JOB.txt

# Combine information into stats
for x in `seq 100`; do 
  cut -d ' ' -f 4- $lats_dir/arc_info/arc_info_sym.$x.txt | \
    steps/dict/get_stats_from_arc_info.py - \
    $lats_dir/arc_info/arc_stats.$x.txt
done

# Combine stats
for x in `seq 100`; do
  cat $lats_dir/arc_info/arc_stats.$x.txt | \
    python steps/dict/get_stats_from_arc_info.py
done | python steps/dict/get_stats_from_arc_info.py | sort -k2,2 -k1,1nr > \
  $lats_dir/arc_info/arc_stats.txt

############################################################################### 
# Create tri3b lexicon and lang
############################################################################### 
cp -r data/local/dict_nosp data/local/dict_learned_nosp/lexiconp.txt
rm data/local/dict_learned_nosp/lexicon.txt
rm data/local/dict_learned_nosp/lexicon_words.txt

# TODO: Use only the best path pronunciation instead of all pronunciations
steps/dict/prons_to_lexicon.py --min-count=0 --max-prons-weight=0.9 \
  --set-max-to-one=true --min-prob=0.3 \
  $lats_dir/arc_info/arc_stats.txt - | sort -k1,1 -k2,2n - | \
  utils/apply_map.pl -f 3- $debug_dir/phone_map.txt | \
  steps/dict/merge_lexicon.py - - data/local/dict_extended_nosp/lexiconp.txt | \
  sort -k1,1 -k2,2n > data/local/dict_learned_nosp/lexiconp.txt

utils/prepare_lang.sh data/local/dict_learned_nosp "<unk>" data/{local/,}lang_learned_nosp 
steps/align_fmllr.sh --cmd "$train_cmd" --nj 100 \
  data/train data/lang_learned_nosp exp/tri3 exp/tri3_learned_ali
steps/train_sat.sh --cmd "$train_cmd" 5000 10000 data/train \
  data/lang_learned_nosp exp/tri2_ali exp/tri3_learned_lex
 utils/mkgraph.sh data/lang_learned_nosp_test/ exp/tri3_learned_lex{,graph_learned}
steps/decode_fmllr.sh --nj 40 --cmd "$decode_cmd" --num-threads 4 \
  exp/tri3_learned_lex/graph data/dev exp/tri3_learned_lex/decode_dev

