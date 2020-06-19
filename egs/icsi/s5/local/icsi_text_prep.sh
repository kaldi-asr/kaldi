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

meet=$(head -n1 $outdir/meetlist)
[ ! -f "$icsi_trans_dir/transcripts/$meet.mrt" ] \
  && echo "$0. $meet.mrt expected to exists, make sure $icsi_trans_dir/transcripts/$meet.mrt is available. " \
  && exit 1;

perl -e 'use XML::LibXML; use Data::Dumper;' || {
  echo "At least one of the expected perl libs {XML::LibXML, Data::Dumber} is missing. "
  echo "To install, type in cmd line: cpan, then install XML::LibXML"
  exit 1;
}

echo "Extracting meetings...."
#extract easily parsable stuff out of mrt files
rm -f $outdir/all.txt
touch $outdir/all.txt
while read -r line; do
  echo "Parsing meeting $line"
  local/icsi_parse_transcripts.pl $icsi_trans_dir/transcripts/$line.mrt $outdir/$line.mrt.txt
  cat $outdir/$line.mrt.txt >> $outdir/all.txt
done < $outdir/meetlist;

echo "Normalising data"
#normalise transcripts
cat $outdir/all.txt | local/icsi_normalise_segments.pl > $outdir/all_normalised.txt
# perfrom some dict matching e.g. LIVING-ROOM -> LIVINGROOM (if the former does not exist, 
# but the latter is in the dictionary
local/icsi_agree_words.sh $outdir/all_normalised.txt data/local/dict/lexicon.txt $outdir/match_with_dict

[ ! -f $outdir/match_with_dict/segments2 ] && \
  echo "Dict matching failed..." && exit 1;
cp $outdir/match_with_dict/segments2 $outdir/all_final.txt

echo "Preparing final train/dev/eval splits"
# make final train/dev/eval splits
for dset in train eval dev; do
  grep -f local/split_$dset.$ext $outdir/all_final.txt > $outdir/$dset.txt
done

echo "ICSI text preparation succeeded"
