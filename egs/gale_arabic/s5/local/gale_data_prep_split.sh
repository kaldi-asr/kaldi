#!/bin/bash 

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0

if [ $# -ne 1 ]; then
   echo "Arguments should be the <gale folder>"; exit 1
fi


#data will data/local

galeData=$(utils/make_absolute.sh $1)
mkdir -p data/local
dir=$(utils/make_absolute.sh data/local)


grep -f local/test_list $galeData/all | grep -v -f local/bad_segments > $galeData/all.test
grep -v -f local/test_list $galeData/all | grep -v -f local/bad_segments > $galeData/all.train 

for x in test train; do
 outdir=$dir/$x
 file=$galeData/all.$x 
 mkdir -p $outdir
 awk '{print $2 " " $2}' $file | sort -u > $outdir/utt2spk 
 cp -pr $outdir/utt2spk $outdir/spk2utt
 awk '{print $2 " " $1 " " $3 " " $4}' $file  | sort -u > $outdir/segments
 awk '{printf $2 " "; for (i=5; i<=NF; i++) {printf $i " "} printf "\n"}' $file | sort -u > $outdir/text
done 


grep -f local/test_list $galeData/wav.scp > $dir/test/wav.scp

cat $galeData/wav.scp | awk -v seg=$dir/train/segments 'BEGIN{while((getline<seg) >0) {seen[$2]=1;}}
 {if (seen[$1]) { print $0}}' > $dir/train/wav.scp
 
echo data prep split succeeded
