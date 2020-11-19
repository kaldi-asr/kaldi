#!/usr/bin/env bash
# 2020 author Jiayu DU

# This script uses kenlm to estimate an arpa model from plain text,
# it is a resort when you hit memory limit dealing with large scale training corpus
# kenlm estimates arpa using on-disk structure,
# as long as you have big enough hard disk, memory shouldn't be a problem.
# by default, kenlm use up to 50% of your local memory, you can control this through -S option

[ -f path.sh ] && . ./path.sh;

kenlm_opts="" # e.g. "-o 4 -S 50% --prune 0 5 7 7"

if [ $# != 4 ]; then
  echo "$0 <text> <kaldi_symbol_table> <working_dir> <arpa_name>"
  echo "e.g. $0 train.txt words.txt wdir 4gram"
  exit 1
fi

text=$1
symbol_table=$2
dir=$3
arpa_name=$4

if ! which lmplz >& /dev/null ; then
  echo "$0: cannot find training tool *lmplz*."
  echo "The KenLM module installed in Kaldi(via tools/extras/install_kenlm_query_only.sh), only supports runtime query,"
  echo "to *train* an arpa using KenLM, you need a complete KenLM installation(which depends on EIGEN and BOOST),"
  echo "you should follow KenLM's cmake building instructions at (https://github.com/kpu/kenlm)"
  exit 1
fi

# the text should be properly pre-processed(cleand, normalized and possibly word-segmented in some languages)

# get rid off irrelavent symbols, the rest of symbols are used as LM training vocabulary. 
grep -v '<eps>' $symbol_table | grep -v '#0' | grep -v '<unk>' | grep -v '<UNK>' | grep -v '<s>' | grep -v '</s>' | awk '{print $1}' > $dir/ngram.vocab

# To make sure that kenlm & kaldi have strictly the same vocabulary:
# 1. feed vocabulary into kenlm via --limit_vocab_file
# 2. cat vocabulary to training text, so each word at least appear once
# 
# TL;DR reason:
# Unlike SRILM's -limit-vocab, kenlm's --limit_vocab_file option 
# spcifies a *valid* set of vocabulary, whereas *valid but unseen* 
# words are discarded in final arpa.
# So the trick is, 
# we explicitly add kaldi's vocab(one word per line) to training text, 
# making each word appear at least once.
# kenlm never prunes unigram, so this always generates consistent kenlm vocabuary as kaldi has.
# The effect of this is like add-one smoothing to unigram counts,
# shouldn't have significant impacts in practice.
cat $dir/ngram.vocab $text | lmplz $kenlm_opts --limit_vocab_file $dir/ngram.vocab > $dir/${arpa_name}.arpa

echo "$0: Done training arpa to: $dir/${arpa_name}.arpa"
