#!/bin/bash

tag_percentage=0.1
word2baseform=data/local/wordlist.txt
lexicon=data/local/lexicon.txt
morfessor=false
unk="<unk>"

. ./utils/parse_options.sh

train_data_dir=`utils/make_absolute.sh ./data/raw_train_data`

# ----------- Collect all words and their counts from the transcripts ---------------------
echo "-------------------------------"
echo " Creating word list and word counts"
echo "-------------------------------"
local/lexicon/make_word_list.py $train_data_dir/filelist.list $train_data_dir/transcription data/local/wordcounts.txt 
awk '{print $2}' data/local/wordcounts.txt > data/local/wordlist.txt


# ------------------------------------------------------------------
# Create 2 special mini "lexicons" for the silence words and any extra 
# words whose pronunciations we may already have
# -----------------------------------------------------------------
echo "---------------------------------"
echo " Making lexical entries for extra words not present in the word list"
echo "----------------------------------"
echo -e "<silence> SIL\n<unk> <oov>\n<noise> <sss>\n<v-noise> <vns>" > data/local/silence_lexicon.txt
echo -e "<hes> <hes>" > data/local/extra_lexicon.txt


# -----------------------------------------------------------------
# Train morphemes using morphessor
# -----------------------------------------------------------------
if $morfessor; then
  local/lexicon/train_morphs.sh data/local/wordcounts.txt data/local/morphs
  local/lexicon/apply_morphs.sh data/local/morphs data/local/wordlist.txt \
     data/local/morphs/word2baseform.txt

  word2baseform=data/local/morphs/word2baseform.txt
fi


# -------------------------------------------------------------
# Create the actual kaldi format lexicon (lexicon.txt) 1 word pronunciation
# pair per line
# -------------------------------------------------------------
echo "---------------------------------------"
echo " Creating lexicon.txt from the word list (or baseforms provided)"
echo "---------------------------------------"
lexicon_dir=`dirname $lexicon`
local/lexicon/make_unicode_lexicon.py --tag-percentage $tag_percentage \
  --silence-lexicon data/local/silence_lexicon.txt \
  --extra-lexicon data/local/extra_lexicon.txt \
  --verbose data/local/log.lexicon \
  $word2baseform $lexicon $lexicon_dir/grapheme_map.txt


# --------------------------------------------------------------
# Create the rest of the dictionary directory from a lexicon.
# If there are silence words in the lexicon other than SIL, provide those
# as well (e.g. data/local/silence_lexicon.txt)
# --------------------------------------------------------------
echo "------------------------------------------------"
echo " Creating the rest of the dictionary directory from lexicon.txt"
echo "------------------------------------------------"
local/prepare_unicode_lexicon.py --silence-lexicon data/local/silence_lexicon.txt \
  $lexicon data/local 


# ------------------------------------------------------------
# Expand the graphemic lexicon from a list of new words.
# Assuming data/local/new_words.txt exists.
# ------------------------------------------------------------
echo "------------------------------------------------"
echo " Adding new words to the lexicon"
echo "------------------------------------------------"
oovs2baseform=data/local/oovs.txt

[ ! -f data/local/new_words.txt ] && echo "No file data/local/new_words.txt" && exit 1

comm -13 <(sort data/local/wordlist.txt) <(sort -u data/local/new_words.txt) > $oovs2baseform

if $morfessor; then
  local/lexicon/apply_morphs.sh data/local/morphs $oovs2baseform \
     data/local/morphs/oovs2baseform.txt

  oovs2baseform=data/local/morphs/oovs2baseform.txt
fi

local/lexicon/make_unicode_lexicon.py --apply-map data/local/grapheme_map.txt \
  --extra-lexicon data/local/lexicon.txt \
  $oovs2baseform data/local/lexicon_expanded.txt data/local/grapheme_map_expanded.txt


mkdir -p data/dict_expanded
cp data/local/{nonsilence_phones,silence_phones,extra_questions,optional_silence}.txt data/dict_expanded/
cp data/local/lexicon_expanded.txt data/dict_expanded/lexicon.txt


# ------------------------------------------------------------------
# Prepare the lang directory corresponding to the expanded lexicon
# ------------------------------------------------------------------
echo "--------------------------------------"
echo " Preparing the updated lang directory "
echo "--------------------------------------"
lang_orig=data/lang
utils/prepare_lang.sh  --phone-symbol-table $lang_orig/phones.txt \
  --share-silence-phones true \
  data/dict_expanded "$unk" data/dict_expanded/lang.tmp data/lang_expanded

exit 0
