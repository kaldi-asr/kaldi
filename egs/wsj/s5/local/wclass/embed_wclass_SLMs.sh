#!/bin/bash

# Copyright FAU Erlangen-Nuremberg (Author: Axel Horndasch) 2016
#
# This script embeds sub-languages models for all word classes which are
# listed in the file $wclass_list (e.g. data/local/wclass/wclass_list.txt).
#
# It is assumed that the SLMs have been created (e.g. by
# create_wclass_SLMs.sh), that they have been converted to FSTs (named
# <word-class-label>.fst) and that the according word class labels (e.g.
# C=CITYNAME) are entries in $all_words_txt (this is words.txt + the necessary
# non-terminal tokens which which are replaced using fstreplace).
#
# The original G.fst is replaced with the new grammar/language model.

set -e

echo "$0 $@"  # Print the command line for logging
. ./path.sh
. utils/parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "usage: embed_wclass_SLMs.sh <wclass-dir> <lm-fst-dir>" && exit 1;
fi

wclass_dir=$1
lm_fst_dir=$2

# Files/directories which are assumed to exist already
wclass_list=$wclass_dir/wclass_list.txt
wclass_lm_dir=$wclass_dir/lm
all_words_txt=$wclass_dir/all_words.txt


g_fst=$lm_fst_dir/G.fst
g_wclass_fst=$lm_fst_dir/G_wclass.fst
g_tmp_fst=$lm_fst_dir/G_tmp.fst

cp $g_fst $g_wclass_fst

# Go through all word classes in 'wclass_list.txt' and
# embed all word-class-based sub-language models in the
# parent G.fst (by replacing the word class labels in
# G.fst using fstreplace).
for wclass in $( cat $wclass_list | awk '{ print $1 }' | sed 's/^C=//' ); do
  wclass_label="C=$wclass"
  # example: if word class 'CITYNAME' is used, we now have
  # wclass='CITYNAME' and wclass_label='C=CITYNAME'

  echo "Replacing label $wclass_label with sub-language model for word class $wclass ..."
  # In words.txt the word class label is mapped to an integer, we need that number.
  # e.g. from "C=WEEKDAY 15841", we extract "15841"
  wclass_word_id=`awk -v current_word=$wclass_label '{ if ( $1 == current_word ) { print $2 } }' $all_words_txt`
  echo "Extracted word ID for word class $wclass: $wclass_word_id"

  # Embedding of sub-language models in the overall language model
  if [ -n "$wclass_word_id" ]; then
    fstreplace --epsilon_on_replace $g_wclass_fst -1 $wclass_lm_dir/$wclass.fst $wclass_word_id |\
    fstrmepsilon |\
    fstminimizeencoded |\
    fstarcsort --sort_type=ilabel > $g_tmp_fst
    mv $g_tmp_fst $g_wclass_fst
  else
    echo Could not extract word ID for $wclass_label, exiting... && exit 1;
  fi
done

# The old G is now overwritten with the new G with embedded
# word-class sub-language models.
mv $g_wclass_fst $g_fst

exit 0;
