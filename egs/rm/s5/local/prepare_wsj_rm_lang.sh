#!/bin/bash
# Copyright 2017 Pegah Ghahremani

# This script prepares a dictionary for wsj-rm experiment using wsj phone set, lexicon and dict and
# rm's words.txt are copied from wsj lexicon for common words in wsj
# and rm. words in rm that are not available in the wsj lexicon are added
# as oov in lexicon.txt.
# The oov word "<SPOKEN_NOISE>" in wsj is also added to words.txt and G.fst is recompiled using
# updated word list.

if [ -f path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: local/prepare_wsj_rm_lang.sh <src-dict> <src-lang> <src-tg-lang-dir>"
  echo "e.g:"
  echo "$0 ../../wsj/s5/data/local/dict ../../wsj/s5/data/lang_nosp data/wsj_rm_dir"
fi

src_dict=$1
src_lang=$2
src_tgt_lang=$3

required_dict_files="$src_dict/lexicon.txt $src_dict/nonsilence_phones.txt $src_dict/silence_phones.txt $src_dict/optional_silence.txt $src_lang/oov.txt $src_lang/phones.txt"
for f in $required_dict_files; do
  if [ ! -f $f ]; then
    echo "file $f that is required for preparing lang does not exists." && exit 1;
  fi
done

rm -rf $src_tgt_lang
mkdir -p $src_tgt_lang
mkdir -p $src_tgt_lang/local
# copy *phones.txt from source to target.
cp -r $src_dict $src_tgt_lang/local/dict
rm $src_tgt_lang/local/dict/lexicon*.txt

oov_word=`cat $src_lang/oov.txt`
# common word list in rm lexicon with lexicon in wsj
comm -12 <(awk '{print $1}' data/local/dict/lexicon.txt | sed "s/\+/\'/g" | sort) \
<(awk '{print $1}' $src_dict/lexicon.txt | sort) | \
sed -r "s/'/+/g" | sort > $src_tgt_lang/words_tmp.txt

comm -23 <(awk '{print $1}' data/local/dict/lexicon.txt | sed "s/\+/\'/g" | sort) \
<(awk '{print $1}' $src_dict/lexicon.txt | sort) | \
sed -r "s/'/+/g" | sort > $src_tgt_lang/words_only_tgt.txt

# add <SPOKEN_NOISE> to rm_swj_word list
(echo "$oov_word"; cat $src_tgt_lang/words_tmp.txt) | sort > $src_tgt_lang/words_tgt_src.txt
rm $src_tgt_lang/words_tmp.txt

# we use wsj lexicon and find common word list in rm and wsj to generate lexicon for rm-wsj
# using wsj phone sets. More than 90% of words in RM are in WSJ(950/994).
cat $src_tgt_lang/words_tgt_src.txt | sed "s/\+/\'/g" | \
utils/apply_map.pl --permissive $src_dict/lexicon.txt | \
paste <(cat $src_tgt_lang/words_tgt_src.txt) - > $src_tgt_lang/local/dict/lexicon_tgt_src.txt

# extend lexicon.txt by adding only_tg words as oov.
oov_phone=`grep "$oov_word" $src_dict/lexicon.txt | cut -d' ' -f2`
cat $src_tgt_lang/local/dict/lexicon_tgt_src.txt <(sed 's/$/ SPN/g' $src_tgt_lang/words_only_tgt.txt) | sort -u > $src_tgt_lang/local/dict/lexicon.txt

# prepare dictionary using new lexicon.txt for RM-SWJ.
utils/prepare_lang.sh --phone-symbol-table $src_lang/phones.txt \
$src_tgt_lang/local/dict "$oov_word" $src_tgt_lang/local/lang_tmp $src_tgt_lang

# Generate new G.fst using updated words list with added <SPOKEN_NOISE>
fstcompile --isymbols=$src_tgt_lang/words.txt --osymbols=$src_tgt_lang/words.txt --keep_isymbols=false \
    -keep_osymbols=false data/local/tmp/G.txt | fstarcsort --sort_type=ilabel > $src_tgt_lang/G.fst || exit 1;
