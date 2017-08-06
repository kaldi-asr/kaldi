#!/bin/bash

###########################################################################################
# This script was copied from egs/ami/s5/local/ami_text_prep.sh
# The source commit was e69198c3dc5633f98eb88e1cdf20b2521a598f21
# Changes made:
#  - Modified paths to match multi_en naming conventions
###########################################################################################

# Copyright 2015, Brno University of Technology (Author: Karel Vesely)
# Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski), 2014, Apache 2.0

if [ $# -ne 1 ]; then
  echo "Usage: $0 <ami-dir>"
  echo " <ami-dir> is download space."
  exit 1;
fi

set -eux

dir=$1
mkdir -p $dir

echo "Downloading annotations..."

amiurl=http://groups.inf.ed.ac.uk/ami
annotver=ami_public_manual_1.6.1
annot="$dir/$annotver"

logdir=data/local/ami/downloads; mkdir -p $logdir/log
[ ! -f $annot.zip ] && wget -nv -O $annot.zip $amiurl/AMICorpusAnnotations/$annotver.zip &> $logdir/log/download_ami_annot.log

if [ ! -d $dir/annotations ]; then
  mkdir -p $dir/annotations
  unzip -o -d $dir/annotations $annot.zip &> /dev/null
fi

[ ! -f "$dir/annotations/AMI-metadata.xml" ] && echo "$0: File AMI-Metadata.xml not found under $dir/annotations." && exit 1;


# extract text from AMI XML annotations,
local/ami_xml2text.sh $dir

wdir=data/local/ami/annotations
[ ! -f $wdir/transcripts1 ] && echo "$0: File $wdir/transcripts1 not found." && exit 1;

echo "Preprocessing transcripts..."
local/ami_split_segments.pl $wdir/transcripts1 $wdir/transcripts2 &> $wdir/log/split_segments.log

# make final train/dev/eval splits
for dset in train eval dev; do
  [ ! -f local/ami_split_$dset.final  ] &&  cp local/ami_split_$dset.orig local/ami_split_$dset.final
  grep -f local/ami_split_$dset.final $wdir/transcripts2 > $wdir/$dset.txt
done


