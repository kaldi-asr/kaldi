#!/bin/bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error
echo "$0 " "$@"

if [ $# -ne 2 ] ; then
  echo "Invalid number of script parameters. "
  echo "  $0 <path-to-material-corpus> <language-name>"
  echo "e.g."
  echo "  $0 /export/corpora5/MATERIAL/IARPA_MATERIAL_BASE-1A-BUILD_v1.0/ swahili"
  exit
fi
data=$1
language=$2

lexicon=$data/conversational/reference_materials/lexicon.txt
[ ! -f $lexicon ] && echo "Lexicon $lexicon does not exist!" && exit 1;

mkdir -p data/$language/local/dict_nosp/
cat data/$language/train/text | cut -f 2- -d ' ' | \
  sed 's/ /\n/g' | sort -u > data/$language/local/dict_nosp/wordlist

local/convert_lexicon.pl <(echo -e "<unk>\t<unk>\n<sil>\t<sil>\n<noise>\t<noise>\n<spnoise>\t<spnoise>" | cat - $lexicon ) data/$language/local/dict_nosp/wordlist | sort -u > data/$language/local/dict_nosp/lexicon.txt
[ -f  data/$language/local/dict_nosp/lexiconp.txt ] && rm data/$language/local/dict_nosp/lexiconp.txt

cat data/$language/local/dict_nosp/lexicon.txt | sed 's/\t/ /g' | \
  cut -f 2- -d ' ' | sed 's/ /\n/g' | sort -u > data/$language/local/dict_nosp/phones.txt


grep "^<.*>$" data/$language/local/dict_nosp/phones.txt  > data/$language/local/dict_nosp/silence_phones.txt
grep -v "^<.*>$" data/$language/local/dict_nosp/phones.txt  > data/$language/local/dict_nosp/nonsilence_phones.txt
echo "<sil>" > data/$language/local/dict_nosp/optional_silence.txt
echo "<unk>" > data/$language/local/dict_nosp/oov.txt



utils/validate_dict_dir.pl data/$language/local/dict_nosp/

