#!/bin/bash

# Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski)
# Changed by Yulan Liu for SWC recipe, 3 Mar 2016

. ./path.sh

nj=$1
job=$2
numch=$3
ses=$4
sdir=$5
odir=$6
wdir=data/local/beamforming

set -e
set -u

utils/split_scp.pl -j $nj $((job-1)) $ses $ses.$job

bmitdir=$odir/bmitraw
mkdir -p $bmitdir

while read line; do

  mkdir -p $odir/$line
  BeamformIt -s $line -c $wdir/channels_$numch \
                        --config_file `pwd`/conf/swc_beamformit.cfg \
                        --source_dir $sdir \
                        --result_dir $bmitdir/$line
  mkdir -p $odir/$line

  ses=`echo $line | sed 's/\_seg//g'`
  mv $bmitdir/$line/${line}.del  $bmitdir/$line/${line}_MDM$numch.del
  mv $bmitdir/$line/${line}.del2 $bmitdir/$line/${line}_MDM$numch.del2
  mv $bmitdir/$line/${line}.info $bmitdir/$line/${line}_MDM$numch.info
  mv $bmitdir/$line/${line}.weat $bmitdir/$line/${line}_MDM$numch.weat
  mv $bmitdir/$line/${line}.wav  $odir/${ses}.wav
#  mv $odir/$line/${line}.wav  $odir/$line/${line}_MDM$numch.wav
  #mv $odir/$line/${line}.ovl  $odir/$line/${line}_MDM$numch.ovl # Was not created!

done < $ses.$job

