#!/bin/bash

giga_dir=$1

source_dir=/export/corpora5/LDC/LDC2011T11/arb_gw_5
num=2000000
suffix="2000k"

[ ! -d $source_dir ] && echo "source Arabic Gigaword does not exist." && exit 1;

[ -f $giga_dir/text ] && mv $giga_dir/text $giga_dir/text.bkp
mkdir -p $giga_dir/

find $source_dir/data/ -name "*.gz" | while read file; do
  gunzip -c $file | local/arabic_convert.py - >> $giga_dir/text.arb
done

head -n $num $giga_dir/text.arb > $giga_dir/text.arb.${suffix}
local/normalize_transcript_BW.pl $giga_dir/text.arb.${suffix} $giga_dir/text.${suffix}

echo "finish preparing Arabic Gigaword"
exit 0
