#!/usr/bin/env bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error
echo "$0 " "$@"

language=swahili

. ./utils/parse_options.sh

if [ $# -ne 1 ] ; then
  echo "Invalid number of script parameters. "
  echo "  $0 [options] <path-to-material-corpus>"
  echo "e.g."
  echo "  $0 --language swahili /export/corpora5/MATERIAL/IARPA_MATERIAL_BASE-1A-BUILD_v1.0/"
  exit
fi
data=$1

lexicon=$data/conversational/reference_materials/lexicon.txt

mkdir -p data/local
cat $lexicon | awk '{print $1}' > data/local/lexicon_words
cat $lexicon | cut -f2-  > data/local/lexicon_phns

if [ "$language" == "swahili" ]; then
  language_affix="sw"
elif [ "$language" == "tagalog" ]; then
  language_affix="tl"
elif [ "$language" == "somali" ]; then
  language_affix="so"
fi
MOSES=/home/pkoehn/moses
SOURCE_TC_MODEL=/home/pkoehn/experiment/material-asr-${language_affix}-en/truecaser/truecase-model.1.${language_affix}
  $MOSES/scripts/recaser/truecase.perl -model $SOURCE_TC_MODEL \
    < data/local/lexicon_words > data/local/lexicon_words_tc

paste data/local/lexicon_words_tc data/local/lexicon_phns | sort > data/local/lexicon_tc

lexicon=data/local/lexicon_tc

[ ! -f $lexicon ] && echo "Lexicon $lexicon does not exist!" && exit 1;
echo $0: using lexicon $lexicon
mkdir -p data/local/dict_nosp/
cat data/train/text | cut -f 2- -d ' ' | \
  sed 's/ /\n/g' | grep . | sort -u > data/local/dict_nosp/wordlist

local/convert_lexicon.pl <(echo -e "<unk>\t<unk>\n<sil>\t<sil>\n<noise>\t<noise>\n<spnoise>\t<spnoise>" | cat - $lexicon ) data/local/dict_nosp/wordlist | sort -u > data/local/dict_nosp/lexicon.txt
[ -f  data/local/dict_nosp/lexiconp.txt ] && rm data/local/dict_nosp/lexiconp.txt

cat data/local/dict_nosp/lexicon.txt | sed 's/\t/ /g' | \
  cut -f 2- -d ' ' | sed 's/ /\n/g' | grep . | sort -u > data/local/dict_nosp/phones.txt


grep "^<.*>$" data/local/dict_nosp/phones.txt  > data/local/dict_nosp/silence_phones.txt
grep -v "^<.*>$" data/local/dict_nosp/phones.txt  > data/local/dict_nosp/nonsilence_phones.txt
echo "<sil>" > data/local/dict_nosp/optional_silence.txt
echo "<unk>" > data/local/dict_nosp/oov.txt



utils/validate_dict_dir.pl data/local/dict_nosp/

