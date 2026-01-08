#!/bin/bash

# Adapted from the beamforming script in AMI recipe and improved
# for SWC recipe "s5b".
#
# Copyright 2016, University of Sheffield (Author: Yulan Liu)
# Apache 2.0

wiener_filtering=false
nj=4
cmd=run.pl

# End configuration section
echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Wrong #arguments ($#, expected 4)"
  echo "Usage: steps/swc_beamform.sh [options] <num-mics> <mic-group> <swc-dir> <wav-out-dir>"
  echo "main options (for others, see top of script file)"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --wiener-filtering <true/false>          # Cancel noise with Wiener filter prior to beamforming"
  exit 1;
fi

#   local/swc_beamform.sh --cmd "$train_cmd" --nj 20 $nmics $ch $SWCDIR $MDMDIR
numch=$1
ch=$2
sdir=$3
odir=$4

wdir=data/local/beamforming

set -e 
set -u

mkdir -p $odir
mkdir -p $wdir/log

[ -e $odir/.done_beamforming_${numch}_${ch} ] && echo "Beamforming already done, skipping..." && exit 0

meetings=$wdir/meetings.list
inwav1=$wdir/inwav1.list
ses=$wdir/ses.list

ls $SWCDIR/swc?/audio/*/SWC*${ch}*.wav | grep $CH"-01" | sed 's/\-01\.wav//g' | awk 'BEGIN{FS="/"}{print $(NF-3)"/"$(NF-2)"/"$(NF-1)"/"$NF}' | sort -u > $inwav1
awk 'BEGIN{FS="/"}{print $NF}' $inwav1 | awk 'BEGIN{FS="_"}{print $1}' > $ses
paste $ses $inwav1 > $meetings

ch_inc=$((8/$numch))
bmf=
for ch in `seq 1 $ch_inc 8`; do
  bmf="$bmf $ch"
done

echo "Will use the following channels: $bmf"

# make the channel file,
if [ -f $wdir/channels_$numch ]; then
  rm $wdir/channels_$numch_$ch
fi
touch $wdir/channels_$numch_$ch


# SWC1-00001_TBL1 SWC1-00001_TBL1-01.sph SWC1-00001_TBL1-02.sph SWC1-00001_TBL1-03.sph SWC1-00001_TBL1-04.sph SWC1-00001_TBL1-05.sph SWC1-00001_TBL1-06.sph SWC1-00001_TBL1-07.sph SWC1-00001_TBL1-08.sph
while read line;
do
  channels=`echo $line | awk '{print $1}'`
  pre=`echo $line | awk '{print $2}'`
  for ch in $bmf; do
    channels="$channels  ${pre}-0$ch.wav"
  done
  echo $channels >> $wdir/channels_$numch
done < $meetings


# do noise cancellation,
if [ $wiener_filtering == "true" ]; then
  echo "Wiener filtering not yet implemented."
  exit 1;
fi


# do beamforming,
echo -e "Beamforming\n"
$cmd JOB=1:$nj $wdir/log/beamform.JOB.log \
     local/beamformit.sh $nj JOB $numch $ses $sdir $odir

touch $odir/.done_beamforming_$numch_$ch





