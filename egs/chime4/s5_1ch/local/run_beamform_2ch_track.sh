#!/bin/bash

# Copyright 2015, Mitsubishi Electric Research Laboratories, MERL (Author: Shinji Watanabe)

. ./cmd.sh
. ./path.sh

# Config:
nj=10
cmd=run.pl

. utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Wrong #arguments ($#, expected 3)"
   echo "Usage: local/run_beamform_2ch_track.sh [options] <wav-in-dir> <wav-out-dir>"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   exit 1;
fi

sdir=$1
odir=$2

wdir=data/beamforming_2ch_track

if [ -z $BEAMFORMIT ] ; then
  export BEAMFORMIT=$KALDI_ROOT/tools/extras/BeamformIt
fi
export PATH=${PATH}:$BEAMFORMIT
! hash BeamformIt && echo "Missing BeamformIt, run 'cd ../../../tools/; extras/install_beamformit.sh;'" && exit 1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

mkdir -p $odir
mkdir -p $wdir/log

allwavs=`find $sdir/ | grep "\.wav" | tr ' ' '\n' | awk -F '/' '{print $(NF-1)"/"$NF}'`

# wavfiles.list can be used as the name of the output files
output_wavfiles=$wdir/wavfiles.list
echo $allwavs | tr ' ' '\n' | awk -F '.' '{print $1}' | sort | uniq > $output_wavfiles

# channel list
input_arrays=$wdir/channels
echo $allwavs | tr ' ' '\n' | sort | awk 'NR%2==1' > $wdir/channels.1st
echo $allwavs | tr ' ' '\n' | sort | awk 'NR%2==0' > $wdir/channels.2nd
paste -d" " $output_wavfiles $wdir/channels.1st $wdir/channels.2nd > $input_arrays

# split the list for parallel processing
split_wavfiles=""
for n in `seq $nj`; do
  split_wavfiles="$split_wavfiles $output_wavfiles.$n"
done
utils/split_scp.pl $output_wavfiles $split_wavfiles || exit 1;

echo -e "Beamforming\n"
# making a shell script for each job
for n in `seq $nj`; do
cat << EOF > $wdir/log/beamform.$n.sh
while read line; do
  $BEAMFORMIT/BeamformIt -s \$line -c $input_arrays \
    --config_file `pwd`/conf/chime4.cfg \
    --source_dir $sdir \
    --result_dir $odir
done < $output_wavfiles.$n
EOF
done
# making a subdirectory for the output wav files
for x in `awk -F '/' '{print $1}' $output_wavfiles | sort | uniq`; do
  mkdir -p $odir/$x
done

chmod a+x $wdir/log/beamform.*.sh
$cmd JOB=1:$nj $wdir/log/beamform.JOB.log \
  $wdir/log/beamform.JOB.sh

echo "`basename $0` Done."
