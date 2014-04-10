#!/bin/bash

#tmp1=`pwd`/anot.tmp

cat $1 | perl -pe 's/\[.+?\]/ /g' | \
perl -pe 's/\{.+?\}/ /g' | \
perl -pe 's/ .*?\* / /g' | \
perl -pe 's/ \*.*? / /g' | \
perl -pe 's/\*\*+/ /g' | \
perl -pe 's/ \*[,:.;] / /g' | \
perl -pe 's/ \* / /g' | \
perl -pe 's/\xEF\xBB\xBF/\n/g' | \

perl -pe 's/\[.+?\]/ /g' | \
perl -pe 's/\{.+?\}/ /g' | \
perl -pe 's/ [a-sæøåA-ZÆØÅ0-9]+?\* / /g' | \
perl -pe 's/ \*[a-zæøåA-ZÆØÅ0-9]+? / /g' | \
perl -pe 's/ \*+ / /g' | \
perl -pe 's/ \*[,:.;] / /g' | \
perl -pe 's/\*/ /g' | \
perl -pe 's/\<[P|p]\>/ /g' | \
tr -s ' '

#rm $tmp1
