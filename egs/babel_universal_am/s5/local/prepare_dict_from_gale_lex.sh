#!/bin/bash
. ./path.sh

. ./utils/parse_options.sh

if [ $# -eq 0 ]; then
  echo "Usage: ./local/prepare_dict_from_hkust_lex.sh <idict> <odict>"
  exit 1
fi

idict=$1
odict=$2

# Check that raw lexicon and silence_lexicon.txt exist
for f in lexicon.raw silence_lexicon.txt; do
  [ ! -f ${idict}/${f} ] && echo "$0: Expects $f to exist." && exit 1
done

# Check that phone mappings for arpa2xsampa and silence phones exist
for f in gale-arabic-grapheme-to-xsampa-phones.conf; do
  [ ! -f conf/${f} ] && echo "$0: Expects map $f to exist in conf." && exit 1
done

mkdir -p $odict

# Make dictionary
unset LC_ALL

# This deals with tone and phone conversion
cat ${idict}/lexicon.raw |\
utils/apply_map.pl -f 2- --permissive \
  conf/gale-arabic-grapheme-to-xsampa-phones.conf 2>/dev/null | tee ${odict}/gale-to-xsampa.log |\
sort -u > ${odict}/lexicon.txt

# I'd like to keep the sort order of the output lexicon as it was in the input
awk '(NR==FNR){a[$1]=NR; next}($1 in a){print a[$1], $0}' \
  ${idict}/lexicon.raw ${odict}/lexicon.txt | sort -k 1,1 -n | cut -d' ' -f2- > ${odict}/lexicon.tmp

mv ${odict}/lexicon.tmp ${odict}/lexicon.txt

cp ${idict}/silence_lexicon.txt ${odict}/
# Finally, prepare the rest of the dict
./local/prepare_unicode_lexicon.py --silence-lexicon ${odict}/silence_lexicon.txt \
                                   ${odict}/lexicon.txt $odict

exit 0
