#!/bin/bash
# Copyright (c) 2016, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

set=kwlist
output=data/dev10h.pem/kwset_${set}/

{
  local/search/create_categories.pl $output/keywords.txt
  cat $output/keywords.int | perl -ane '
     if (grep (/^0$/, @F[1..$#F])) {print  "$F[0] OOV=1\n";}
     else { print "$F[0] OOV=0\n";}'
} | local/search/normalize_categories.pl > $output/categories
cut -f 1  data/local/filtered_lexicon.txt | uconv -f utf8 -t utf8 -x Any-Lower | sort -u | \
      nl | awk '{print $2, $1;}' > data/dev10h.pem/kwset_${set}/base_words.txt
    paste <(cut -f 1  data/dev10h.pem/kwset_${set}/keywords.txt ) \
          <(cut -f 2  data/dev10h.pem/kwset_${set}/keywords.txt | \
        uconv -f utf8 -t utf8 -x Any-Lower ) | \
        local/kwords2indices.pl --map-oov 0 data/dev10h.pem/kwset_${set}/base_words.txt |\
      perl -ane '
        if (grep (/^0$/, @F[1..$#F])) {print  "$F[0] BaseOOV=1\n";}
        else { print "$F[0] BaseOOV=0\n";}' |\
      cat - data/dev10h.pem/kwset_${set}/categories | sort -u |\
      local/search/normalize_categories.pl > data/dev10h.pem/kwset_${set}/categories.2
      mv data/dev10h.pem/kwset_${set}/categories data/dev10h.pem/kwset_${set}/categories.bak
      mv data/dev10h.pem/kwset_${set}/categories.2 data/dev10h.pem/kwset_${set}/categories

cp data/dev10h.pem/kwset_kwlist/categories data/dev10h.phn.pem/kwset_kwlist/categories
cp data/dev10h.pem/kwset_kwlist/categories data/dev10h.syll.pem/kwset_kwlist/categories
find exp/ -name ".done.kwset.kwlist" | xargs rm

