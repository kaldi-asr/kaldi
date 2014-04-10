#!/bin/bash

# Copyright 2009-2012 AT&T Labs Research   Copenhagen Business School   (Author: Andreas Kirkedal)
# Apache 2.0.


if [ $# -le 1 ]; then
   echo "Arguments should be a file with a list of text files and a filename for the output."
   exit 1;
fi


flist=$1
$dir=$(dirname $flist)
fout=$2

split -l 50000 $flist $dir/templist_

for f in $dir/templist_*; do 
    cat $f | while read l; do
	cat $l; 
    done > $f.sents;
done

cat $dir/templist_*.sents > $fout

wait

rm -f templist_*

