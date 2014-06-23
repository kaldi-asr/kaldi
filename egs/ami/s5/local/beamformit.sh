#!/bin/bash

# Copyright 2014, University of Edibnurgh (Author: Pawel Swietojanski)

. ./path.sh

nj=$1
job=$2
numch=$3
meetings=$4
sdir=$5
odir=$6
wdir=data/local/beamforming

utils/split_scp.pl -j $nj $job $meetings $meetings.$job

while read line; do

#                        --config_file=`pwd`/conf/beamformit.cfg \
  BeamformIt -s $line -c $wdir/channels_$numch \
                        --source_dir=$sdir \
                        --result_dir=$odir/temp_dir \
                        --do_compute_reference=1
  mkdir -p $odir/$line
  mv $odir/temp_dir/$line/${line}_seg.del  $odir/$line/${line}_MDM$numch.del
  mv $odir/temp_dir/$line/${line}_seg.del2 $odir/$line/${line}_MDM$numch.del2
  mv $odir/temp_dir/$line/${line}_seg.info $odir/$line/${line}_MDM$numch.info
  mv $odir/temp_dir/$line/${line}_seg.ovl  $odir/$line/${line}_MDM$numch.ovl
  mv $odir/temp_dir/$line/${line}_seg.weat $odir/$line/${line}_MDM$numch.weat
  mv $odir/temp_dir/$line/${line}_seg.wa*  $odir/$line/${line}_MDM$numch.wav
  mv $odir/temp_dir/$line/${line}_seg2.wa* $odir/$line/${line}_MDM${numch}_seg2.wav

  rm -r $odir/temp_dir

done < $meetings.$job

