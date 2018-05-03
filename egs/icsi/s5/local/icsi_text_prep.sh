#!/bin/bash

# Pawel Swietojanski
# Apache 2.0

icsi_trans_dir=$1
outdir=$2

if [ $# -ne 2 ]; then
    echo "$0. Usage <icsi-transctiption-dir> <out-dir>"
    exit 1;
fi

mkdir -p $outdir

ext=orig
[ -f local/split_train.final ] && ext=final
cat local/split_*.$ext | sort > $outdir/meetlist

meet=`head -n1 $outdir/meetlist`
[ ! -f "$icsi_trans_dir/transcripts/$meet.mrt" ] \
  && echo "$0. $meet.mrt expected to exists, make sure $icsi_trans_dir/transcripts/$meet.mrt is available. " \
  && exit 1;

#extract easily parsable stuff out of mrt files
rm -f $outdir/all.txt
touch $outdir/all.txt
while read line; do
  echo "Parsing $line"
  local/icsi_parse_transcripts.pl $icsi_trans_dir/transcripts/$line.mrt $outdir/$line.mrt.txt
  cat $outdir/$line.mrt.txt >> $outdir/all.txt
done < $outdir/meetlist;

# make final train/dev/eval splits
for dset in train eval dev; do
  grep -f local/split_$dset.$ext $outdir/all.txt > $outdir/$dset.txt
done

