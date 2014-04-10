#!/bin/bash

#tmp1=`pwd`/anot.tmp

cat $1 | perl -pe 's/[\.|\!|\?]$/ \./g' | \
perl -pe 's/[\.|\!|\?] (_N[LS]_)$/ \. \1/g' | \
perl -pe 's/([:])([ |\n])/ \1\2/g' | \
perl -C -pe 's/["\t´\(\)\[\]\{\}]/ /g' | \
perl -C -pe "s/’/'/g" | \
perl -C -pe 's/[-,-] / /g' | \
perl -C -pe 's/([a-zæøå])-([A-ZÆØÅ])/\1 \2/g' | \
perl -C -pe 's/([A-ZÆØÅ])-([a-zæøå])/\1 \2/g' | \
perl -C -pe 's/[ |\.]-/ /g' | \
perl -C -pe 's/[_\.\+][_\.\+]+/ /g' | \
perl -pe 's/»|«|̣̣©|“|”|;//g' | \
perl -pe 's/([0-3][0-9][0-1][0-9][0-9][0-9])-([0-9][0-9][0-9][0-9])/\1_\2/g' | \
perl -pe 's/([0-2][0-9])[\.|\\]([0-5][0-9])/\1 \2/g' | \
perl -pe 's/([%])/ PROCENT /g' | \
perl -pe 's/([+])/ PLUS /g' | \
perl -pe 's/^- //g' | \
perl -pe 's/^ *| *$//g' | \

tr -s ' '
