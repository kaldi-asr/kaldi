#!/usr/bin/env bash



dir=$1

split -l 50000 $dir/lmtxtfiles $dir/templist_

for f in $dir/templist_*; do
    cat $f | while read l; do
    cat $l;
    done > $f.sents;
done

cat $dir/templist_*.sents > $dir/lmsents

wait

rm -f $dir/templist_*
