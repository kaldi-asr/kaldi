#!/bin/bash
# Copyright 2017 Pegah Ghahremani

# This script prepares a dictionary for sourc-target using source phone set, lexicon and dict and
# target words.txt are copied from source lexicon for common words in source
# and target. words in target that are not available in the source lexicon are added
# as oov in lexicon.txt.
# The <SPOKEN_NOISE> is also added to words.txt and G.fst is recompiled using
# updated word list.

. utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: local/prepare_wsj_rm_lang.sh <src-dict> <src-phones> <src-tg-lang-dir>"
  echo "e.g:"
  echo "$0 ../../wsj/s5/data/local/dict ../../wsj/s5/data/lang/phones.txt data/wsj_rm_dir"
fi

src_dict=$1
src_phones=$2
src_tg_lang=$3

required_dict_files="lexicon.txt nonsilence_phones.txt silence_phones.txt optional_silence.txt"
for f in $required_dict_files; do
  if [ ! -f $src_dict/$f ]; then
    echo "file $src_dict/$f that is required for preparing lang does not exists." && exit 1;
  fi
done

rm -rf $src_tg_lang
mkdir -p $src_tg_lang
mkdir -p $src_tg_lang/local
# copy *phones.txt from source to target.
cp -r $src_dict $src_tg_lang/local/dict
rm $src_tg_lang/local/dict/lexicon.txt

# common word list in rm lexicon with lexicon in wsj
comm -12 <(awk '{print $1}' data/local/dict/lexicon.txt | sed "s/\+/\'/g" | sort) \
<(awk '{print $1}' $src_dict/lexicon.txt | sort) | \
sed -r "s/'/+/g" | sort > $src_tg_lang/words_tmp.txt

comm -23 <(awk '{print $1}' data/local/dict/lexicon.txt | sed "s/\+/\'/g" | sort) \
<(awk '{print $1}' $src_dict/lexicon.txt | sort) | \
sed -r "s/'/+/g" | sort > $src_tg_lang/words_only_tg.txt

# add <SPOKEN_NOISE> to rm_swj_word list
(echo "<SPOKEN_NOISE>"; cat $src_tg_lang/words_tmp.txt) | sort > $src_tg_lang/words_tg_src.txt
rm $src_tg_lang/words_tmp.txt

# we use wsj lexicon and find common word list in rm and wsj to generate lexicon for rm
# using wsj phone sets. More than 90% of words in RM are in WSJ(950/994).
cat $src_tg_lang/words_tg_src.txt | sed "s/\+/\'/g" | \
utils/apply_map.pl --permissive $src_dict/lexicon.txt | \
paste <(cat $src_tg_lang/words_tg_src.txt) - > $src_tg_lang/local/dict/lexicon_tg_src.txt

# extend lexicon.txt by adding only_tg words as oov.
cat $src_tg_lang/local/dict/lexicon_tg_src.txt <(sed 's/$/ SPN/g' $src_tg_lang/words_only_tg.txt) | sort -u > $src_tg_lang/local/dict/lexicon.txt

# prepare dictionary using new lexicon.txt for RM-SWJ.
utils/prepare_lang.sh --phone-symbol-table $src_phones \
$src_tg_lang/local/dict "<SPOKEN_NOISE>" $src_tg_lang/local/lang_tmp $src_tg_lang

# Generate new G.fst using updated words list with added <SPOKEN_NOISE>
fstcompile --isymbols=$src_tg_lang/words.txt --osymbols=$src_tg_lang/words.txt --keep_isymbols=false \
    -keep_osymbols=false data/local/tmp/G.txt | fstarcsort --sort_type=ilabel > $src_tg_lang/G.fst || exit 1;
