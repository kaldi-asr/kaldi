#!/usr/bin/env bash
# Copyright 2017 Pegah Ghahremani

# This script prepares a dictionary for wsj-to-rm transfer learning experiment,
# which uses wsj phone set phones.txt, lexicon lexicon.txt and dict.
# The new lexicon.txt are created for words in rm words.txt as follows:
#   1) The lexicon are copied from wsj lexicon.txt for common words in wsj and rm.
#   2) Words in rm that are not in the wsj lexicon are added
#      as oov to new lexicon.txt.
# The oov word "<SPOKEN_NOISE>" in wsj is also added to words.txt and G.fst is
# recompiled using updated word list.

if [ -f path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: local/prepare_wsj_rm_lang.sh <src-dict> <src-lang> <output-dir>"
  echo "e.g:"
  echo "$0 ../../wsj/s5/data/local/dict ../../wsj/s5/data/lang_nosp data/wsj_rm_dir"
fi

src_dict=$1
src_lang=$2
output_dir=$3

required_dict_files="$src_dict/lexicon.txt $src_dict/nonsilence_phones.txt $src_dict/silence_phones.txt $src_dict/optional_silence.txt $src_lang/oov.txt $src_lang/phones.txt"
for f in $required_dict_files; do
  if [ ! -f $f ]; then
    echo "$0: file $f that is required for preparing lang does not exist." && exit 1;
  fi
done

rm -r $output_dir 2>/dev/null || true
mkdir -p $output_dir
mkdir -p $output_dir/local
# copy *phones.txt from source to target.
cp -r $src_dict $output_dir/local/dict
rm $output_dir/local/dict/lexicon*.txt

oov_word=`cat $src_lang/oov.txt`
# common word list in rm lexicon with lexicon in wsj
comm -12 <(awk '{print $1}' data/local/dict/lexicon.txt | sed "s/\+/\'/g" | sort) \
  <(awk '{print $1}' $src_dict/lexicon.txt | sort) | \
  sed -r "s/'/+/g" | sort > $output_dir/words_tmp.txt

comm -23 <(awk '{print $1}' data/local/dict/lexicon.txt | sed "s/\+/\'/g" | sort) \
  <(awk '{print $1}' $src_dict/lexicon.txt | sort) | \
  sed -r "s/'/+/g" | sort > $output_dir/words_only_tgt.txt

# add oov_word to word list
(echo "$oov_word"; cat $output_dir/words_tmp.txt) | sort > $output_dir/words_tgt_src.txt
rm $output_dir/words_tmp.txt

# we use wsj lexicon and find common word list in rm and wsj to generate lexicon for rm-wsj
# using wsj phone sets. More than 90% of words in RM are in WSJ(950/994).
cat $output_dir/words_tgt_src.txt | sed "s/\+/\'/g" | \
utils/apply_map.pl --permissive $src_dict/lexicon.txt | \
  paste <(cat $output_dir/words_tgt_src.txt) - > $output_dir/local/dict/lexicon_tgt_src.txt

# extend lexicon.txt by adding only_tg words as oov.
oov_phone=`grep "$oov_word" $src_dict/lexicon.txt | cut -d' ' -f2`
cat $output_dir/local/dict/lexicon_tgt_src.txt <(sed 's/$/ SPN/g' $output_dir/words_only_tgt.txt) | sort -u > $output_dir/local/dict/lexicon.txt

# prepare dictionary using new lexicon.txt for RM-SWJ.
utils/prepare_lang.sh --phone-symbol-table $src_lang/phones.txt \
  $output_dir/local/dict "$oov_word" $output_dir/local/lang_tmp $output_dir

# Generate new G.fst using updated words list with added <SPOKEN_NOISE>
fstcompile --isymbols=$output_dir/words.txt --osymbols=$output_dir/words.txt --keep_isymbols=false \
    -keep_osymbols=false data/local/tmp/G.txt | fstarcsort --sort_type=ilabel > $output_dir/G.fst || exit 1;
