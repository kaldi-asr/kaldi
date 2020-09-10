#!/bin/bash

# Copyright 2014, University of Edinburgh (Author: Pawel Swietojanski)
# Copyright 2018, Emotech LTD (Author: Pawel Swietojanski)
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
   echo "Usage: local/icsi_beamform.sh [options] <mic> <icsi-dir> <wav-out-dir>"
   echo "e.g. local/icsi_beamform.sh mdm4 /foo/bar/ICSI /foo/bar/ICSI/beamformed"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --wiener-filtering <true/false>          # Cancel noise with Wiener filter prior to beamforming"
   exit 1;
fi

mic=$1
numch=$(echo $mic | sed 's/[a-z]//g')
sdir=$2
odir=$3
segs=data/local/annotations/all_final.txt
wdir=data/local/beamforming

set -e
set -u

mkdir -p $odir
mkdir -p $wdir/log

[ -e $odir/.done_beamforming_$mic ] && echo "Beamforming already done, skipping..." && exit 0
[ ! -f $segs ] && echo "Expected file $segs to exists, exiting." && exit 1;
[ $numch -lt 1 ] || [ $numch -gt 4 ] && echo "For ICSI mdm channels should be in [1,4], got $numch." && exit 1;

meetings=$wdir/meetings.list

cat local/split_train.orig local/split_dev.orig local/split_eval.orig | sort > $meetings

# generate channel files, note for Bsr001 annotation notes mention two PZM channels were not properly
# ascribed labels during the meeting setting up, but channels exists physically and one can assume 
# those are PZM mics, thus we use them anyway

echo -e "Preparing channel file for $numch channels [1...$numch]....\n"

cat $segs | \
  awk '{ meeting=$1; channel=$2; dchannel=$3; speaker=$4; stime=$5; etime=$6;
         if ( meeting=="Bsr001" ) {
           dchannel=dchannel",chanE,chanF";
         }
         printf("%s %s\n", meeting, dchannel);
       }' | sort -k1 | uniq > $wdir/meet2chans

#agree lower/upper casing in filenames
find $sdir/ -name "*.sph" | sort > $wdir/sph.flist

awk -F'/' -v micdir=$mic '{
      meetid_orig=substr($(NF-1),1,6);
      meetid_norm="B"substr($(NF-1),2,6);
      print meetid_norm" "meetid_orig;
   }' $wdir/sph.flist | sort -k1 | uniq > $wdir/rec2meet

join $wdir/rec2meet $wdir/meet2chans > $wdir/rec2info

# row in rec2info file is Bns001 bns001 chanA,chanB,chanC,chanD 
# if 2nd col starts with 'b' we lowercase the channel names, as
# this is how their corresponding filenames are on disk 
awk -v nmics=$numch \
    '{ meeting=$1; meeting_orig=$2; dchannel=$3;
       if ( meeting_orig ~ /^b/ ) {
         dchannel=tolower(dchannel);
       }
       N=split(dchannel, chans, ",");
       printf("%s", meeting);
       for (i=1; i<=nmics; i++) {
         printf(" %s/%s.sph", meeting_orig, chans[i]);
       }
       printf("\n");
     }' $wdir/rec2info > $wdir/channels_$mic

# do noise cancellation,
if [ $wiener_filtering == "true" ]; then
  echo "Wiener filtering not yet implemented."
  exit 1;
fi

# do beamforming,
echo -e "Beamforming (make take a while...)\n"
$cmd JOB=1:$nj $wdir/log/beamform.JOB.log \
     local/beamformit.sh $nj JOB $mic $meetings $sdir $odir

touch $odir/.done_beamforming_$mic

echo "Beamforming stage with $mic succeeded."
