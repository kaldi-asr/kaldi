#!/bin/bash

#Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski)
#Apache 2.0

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

mkdir -p $odir
mkdir -p $wdir/log

meetings=$wdir/meetings.list

cat local/split_train.orig local/split_dev.orig local/split_eval.orig | sort > $meetings

ch_inc=$((8/$numch))
bmf=
for ch in `seq 1 $ch_inc 8`; do
  bmf="$bmf $ch"
done

echo "Will use the following channels: $bmf"

#make the channel file
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

######
#do beamforming
######

echo -e "Beamforming\n"

$cmd JOB=0:$nj $wdir/log/beamform.JOB.log \
     local/beamformit.sh $nj JOB $numch $meetings $sdir $odir

: << "C"
(

  utils/split_scp.pl -j $nj JOB $meetings $meetings.JOB

  while read line; do
    BeamformIt -s $line -c $wdir/channels_$numch \
                        --config_file=conf/beamformit.cfg \
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
  done < $meetings.JOB

)
C


