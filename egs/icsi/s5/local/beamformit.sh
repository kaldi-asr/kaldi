#!/bin/bash

# Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski)
# Copyright 2018, Emotech LTD (Author: Pawel Swietojanski)
# Apache 2.0

. ./path.sh

nj=$1
job=$2
mic=$3
meetings=$4
sdir=$5
odir=$6
wdir=data/local/beamforming

set -e
set -u

utils/split_scp.pl -j $nj $((job-1)) $meetings $meetings.$job

while read -r line; do

  mkdir -p $odir/$line

  #Pawel: libsndfile does not seem to support shortened sph files, thus
  #converting them first to wav, doing bmf on that, and then removing

  channels_sph=`cat $wdir/channels_$mic | grep $line | cut -f2- -d" "`
  channel_entry="$line"
  for sphfile in $channels_sph; do
    wavfile=`basename $sphfile .sph`".wav"
    echo "Converting $sdir/$sphfile to $odir/$line/$wavfile for beamforming"
    sph2pipe -f wav -p -c 1 $sdir/$sphfile $odir/$line/$wavfile
    channel_entry="$channel_entry $line/$wavfile"
  done

  echo -e "Beamforming: $channel_entry"
  echo "$channel_entry" > $odir/$line/channels

  BeamformIt -s $line -c $odir/$line/channels \
                        --config_file `pwd`/conf/beamformit.cfg \
                        --source_dir $odir \
                        --result_dir $odir/$line

  mv $odir/$line/${line}.del  $odir/$line/${line}_$mic.del
  mv $odir/$line/${line}.del2 $odir/$line/${line}_$mic.del2
  mv $odir/$line/${line}.info $odir/$line/${line}_$mic.info
  mv $odir/$line/${line}.weat $odir/$line/${line}_$mic.weat
  mv $odir/$line/${line}.wav  $odir/$line/${line}_$mic.wav
  #mv $odir/$line/${line}.ovl  $odir/$line/${line}_MDM$numch.ovl # Was not created!

  #remove intermediate files
  rm $odir/$line/chan*.wav

done < $meetings.$job


