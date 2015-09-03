#!/bin/bash

. ./path.sh || die "path.sh expected";

cd data
#convert to FST format for Kaldi
cat local/swahili.arpa | ../utils/find_arpa_oovs.pl lang/words.txt  > lang/oovs.txt
cat local/swahili.arpa |    \
    grep -v '<s> <s>' | \
    grep -v '</s> <s>' | \
    grep -v '</s> </s>' | \
    arpa2fst - | fstprint | \
    ../utils/remove_oovs.pl lang/oovs.txt | \
    ../utils/eps2disambig.pl | ../utils/s2eps.pl | fstcompile --isymbols=lang/words.txt \
      --osymbols=lang/words.txt  --keep_isymbols=false --keep_osymbols=false | \
     fstrmepsilon > lang/G.fst
