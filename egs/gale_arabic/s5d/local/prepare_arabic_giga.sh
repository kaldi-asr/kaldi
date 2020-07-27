#!/usr/bin/env bash

data_dir=/export/corpora/LDC/LDC2011T11/
arabic_giga_dir=Arabic_giga/

[ ! -d $arabic_giga_dir ] && mkdir $arabic_giga_dir

#for x in `find $data_dir -name "*.gz"`; do
#  echo $x
#  dest_file=`basename $x .gz`
#  gunzip -c $x > ${arabic_giga_dir}/${dest_file}.orig
#done

#for x in $arabic_giga_dir/*.orig; do
#  echo "Processing $x"
#  local/arabic_convert.py $x > ${x}.mid
#done

for x in $arabic_giga_dir/*.mid; do
  echo "Processing $x"
  local/normalize_transcript_BW.pl $x ${x}.norm
done
