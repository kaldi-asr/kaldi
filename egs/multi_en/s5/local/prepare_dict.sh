#!/usr/bin/env bash

# Copyright 2017  Intellisist, Inc. (Author: Navneeth K)
#           2017  Xiaohui Zhang
# Apache License 2.0

# This script first prepares switchboard lexicon and CMUDict + tedlium combined lexicon (refered as cmudict later on for simplicity).
# Then it maps phones in switchboard lexicon to cmudict and merge these two lexicons to produce the final lexicon data/local/dict_combined.
# After phone mapping, all alternative pronunciations from switchboard lexicon are included.

replace_swbd_symbols=( "ax" "el" "en" )
replace_cmudict_symbols=( "ah" "ah l" "ah n" )

. ./cmd.sh
. ./path.sh

#check existing directories
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: prepare_dict.sh /path/to/SWBD [/path/to/TEDLIUM_r2]"
  exit 1; 
fi 

SWBD_DIR=$1
TEDLIUM_DIR=$2

# This function filters lines that are common in both files
function filter_common {
    awk 'NR==FNR{arr[$0]++;next} arr[$0] {print}' $1 $2
}

# This function filters lines in file2 that are not in file1
function filter_different {
    awk 'NR==FNR{arr[$0]++;next} !arr[$0] {print}' $1 $2
}

num_syms=0
substitute_arg=""
for i in "${replace_swbd_symbols[@]}"; do
  replace_symbol=${replace_cmudict_symbols[${num_syms}]}
  if [ $num_syms -eq 0 ]; then
    # ax appears twice together in "personably p er s ax ax n b l iy"
    substitute_arg=" sed 's: ${i} : ${replace_symbol} :g' |  sed 's: ${i} : ${replace_symbol} :g' | sed 's:${i}$:${replace_symbol}:g'"
  else
    substitute_arg=$substitute_arg" | sed 's: ${i} : ${replace_symbol} :g' | sed 's:${i}$:${replace_symbol}:g'"
  fi
  num_syms=$((num_syms+1))
done

# Prepare switchboard lexicon
local/swbd1_data_download.sh $SWBD_DIR
local/swbd1_prepare_dict.sh

# Prepare cmudict + tedlium lexicon
local/cmu_tedlium_prepare_dict.sh $TEDLIUM_DIR

dir=data/local/dict_combined
swbd_dir=data/local/dict_swbd
cmudict_dir=data/local/dict_cmu_tedlium

rm -rf $dir && mkdir -p $dir

# Find words that are unique to swbd lexicon (excluding non-scored words)
utils/filter_scp.pl --exclude ${cmudict_dir}/lexicon.txt \
  ${swbd_dir}/lexicon.txt | grep -v '\[*\]' | grep -v '<unk>'  > ${dir}/lexicon_swbd_unique.txt || exit 1;

# Mapping phones from swbd phones to cmu phones for words above.
echo "cat ${dir}/lexicon_swbd_unique.txt | $substitute_arg" > ${dir}/substitute.sh
bash ${dir}/substitute.sh > ${dir}/lexicon_swbd_unique_cmuphones.txt || exit 1;

# Find words that exist in both swbd and cmudict lexicons (excluding non-scored words)
utils/filter_scp.pl --exclude ${dir}/lexicon_swbd_unique.txt \
  ${swbd_dir}/lexicon.txt | grep -v '\[*\]' | grep -v '<unk>' > ${dir}/lexicon_swbd1.txt || exit 1;

# Find words that have same pronounciation in both dictionaries - common lines
filter_common ${cmudict_dir}/lexicon.txt \
  ${dir}/lexicon_swbd1.txt > ${dir}/lexicon_re_match_pron.txt || exit 1;

# Find words in swbd lexicon that have different pronounciation from cmudict - different lines
filter_different ${dir}/lexicon_re_match_pron.txt \
  ${dir}/lexicon_swbd1.txt > ${dir}/lexicon_swbd2.txt || exit 1;

# Mapping phones from swbd phones to cmu phones for words above.
echo "cat ${dir}/lexicon_swbd2.txt | $substitute_arg" > ${dir}/substitute.sh
bash ${dir}/substitute.sh > ${dir}/lexicon_swbd3.txt || exit 1;

# lexicon_re_swbd4.txt contains lines that match after phone mapping
filter_common ${cmudict_dir}/lexicon.txt \
  ${dir}/lexicon_swbd3.txt > ${dir}/lexicon_re_swbd4.txt || exit 1;

# lexicon_swbd4.txt contains lines that do not match after phone mapping (alternative pronunciations).
filter_different ${cmudict_dir}/lexicon.txt \
  ${dir}/lexicon_swbd3.txt > ${dir}/lexicon_swbd4.txt || exit 1;

# Extract lines from cmudict that has the above words
utils/filter_scp.pl ${dir}/lexicon_swbd4.txt ${cmudict_dir}/lexicon.txt > ${dir}/lexicon_cmudict4.txt || exit 1;

# Writing to lexicon.txt
cat ${dir}/lexicon_swbd4.txt ${dir}/lexicon_swbd_unique_cmuphones.txt ${cmudict_dir}/lexicon.txt | sort -u > ${dir}/lexicon.txt

# Separate the lexicon word and phoneme expansion by TAB
cat ${dir}/lexicon.txt | awk '{printf("%s\t",$1); for(i=2;i<NF;i++) {printf("%s ",$i);} printf("%s\n",$NF)}' > ${dir}/lexicon_tab_separated.txt
mv ${dir}/lexicon_tab_separated.txt ${dir}/lexicon.txt

# copy silence, nonsilence and optional silence phones from swbd dict
cp ${cmudict_dir}/{nonsilence_phones.txt,silence_phones.txt,optional_silence.txt,extra_questions.txt} ${dir}

# validate the dict directory
utils/validate_dict_dir.pl $dir
