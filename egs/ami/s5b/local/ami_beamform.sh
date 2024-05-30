#!/usr/bin/env bash

# Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski)
# Apache 2.0

wiener_filtering=false
nj=4
cmd=run.pl

# End configuration section
echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 4)"
   echo "Usage: steps/ami_beamform.sh [options] <num-mics> <ami-dir> <wav-out-dir>"
   echo "e.g. steps/ami_beamform.sh 8 /foo/bar/AMI /foo/bar/AMI/beamformed"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --wiener-filtering <true/false>          # Cancel noise with Wiener filter prior to beamforming"
   exit 1;
fi

numch=$1
sdir=$2
odir=$3
wdir=data/local/beamforming

set -e
set -u

mkdir -p $odir
mkdir -p $wdir/log

[ -e $odir/.done_beamforming ] && echo "Beamforming already done, skipping..." && exit 0

meetings=$wdir/meetings.list

cat local/split_train.orig local/split_dev.orig local/split_eval.orig | sort > $meetings
# Removing ``lost'' MDM session-ids : http://groups.inf.ed.ac.uk/ami/corpus/dataproblems.shtml
mv $meetings{,.orig}; grep -v "IS1003b\|IS1007d" $meetings.orig >$meetings

ch_inc=$((8/$numch))
bmf=
for ch in `seq 1 $ch_inc 8`; do
  bmf="$bmf $ch"
done

echo "Will use the following channels: $bmf"

# make the channel file,
if [ -f $wdir/channels_$numch ]; then
  rm $wdir/channels_$numch
fi
touch $wdir/channels_$numch

while read line;
do
  channels="$line "
  for ch in $bmf; do
    channels="$channels $line/audio/$line.Array1-0$ch.wav"
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
     local/beamformit.sh $nj JOB $numch $meetings $sdir $odir

touch $odir/.done_beamforming
