#! /bin/bash

unk="<unk>"

. ./utils/parse_options.sh

if [ $# -eq 0 ]; then
  echo "Usage: ./local/expand_graphemic_lexicon.sh [--opts] <idict> <ilang> <old_word_list> <new_word_list> <odict> <olang>"
  echo "  options: "
  echo "      --unk \"<unk>\""
  echo
  echo
  echo " Example: "
  echo "   ./local/expand_graphemic_lexicon.sh --unk \"<unk>\" \\"
  echo "      data/local data/lang data/local/word_list.txt data/dict_expand/new_words.txt \\"
  echo "      data/dict_expand data/lang_expand"
  exit 1
fi

idict=$1
ilang=$2
old_words=$3
new_words=$4
odict=$5
olang=$6

# ------------------------------------------------------------
# Expand the graphemic lexicon from a list of new words.
# ------------------------------------------------------------
echo "------------------------------------------------"
echo " Adding new words to the lexicon"
echo "------------------------------------------------"

mkdir -p $odict
oovs2baseform=${odict}/new_oovs.txt

[ ! -f $new_words ] && echo "No file $new_words" && exit 1
[ ! -f $old_words ] && echo "No file $old_words" && exit 1

comm -13 <(sort $old_words) <(sort -u $new_words) > $oovs2baseform

local/lexicon/make_unicode_lexicon.py --apply-map ${idict}/grapheme_map.txt \
  --extra-lexicon ${idict}/lexicon.txt \
  $oovs2baseform ${odict}/lexicon.tmp ${odict}/grapheme_map.txt

# We want to remove words with empty pronunciations. If this occurs at all
# it's probably due to poor text normalization, but we'll deal with it
# regardless.
awk '(NF > 1)' ${odict}/lexicon.tmp > ${odict}/lexicon.txt
rm ${odict}/lexicon.tmp
 
cp $idict/{nonsilence_phones,silence_phones,extra_questions,optional_silence}.txt $odict/

# ------------------------------------------------------------------
# Prepare the lang directory corresponding to the expanded lexicon
# ------------------------------------------------------------------
echo "--------------------------------------"
echo " Preparing the updated lang directory "
echo "--------------------------------------"
utils/prepare_lang.sh  --phone-symbol-table $ilang/phones.txt \
  --share-silence-phones true \
  $odict "$unk" ${odict}/lang.tmp $olang

exit 0
