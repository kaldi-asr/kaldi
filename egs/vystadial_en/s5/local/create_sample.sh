#!/bin/bash
# example usage:
# ./local/create_sample.sh /ha/projects/vystadial/data/asr/en/voip/ Results/vystadial-sample/ test 100
# note that it suppose there are only *.wav and *.wav.trn and the 
# the n is the numbero of files in the directory

src=$1
tgt=$2
typ=$3   # dev/test/train
n=$4

src_dir=$src/$typ
tgt_dir=$tgt/$typ
mkdir -p $tgt_dir
ls $src_dir | head -n $n \
| while read f ; do
    cp $src_dir/$f $tgt_dir
done
