#!/usr/bin/env bash

# Copyright 2015, Mitsubishi Electric Research Laboratories, MERL (Author: Shinji Watanabe)

. ./cmd.sh
. ./path.sh

# Config:
nj=10
cmd=run.pl

. utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Wrong #arguments ($#, expected 2)"
   echo "Usage: local/chime3_beamform.sh [options] <wav-in-dir> <wav-out-dir>"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   exit 1;
fi

sdir=$1
odir=$2
wdir=data/local/beamforming

if [ -z $BEAMFORMIT ] ; then
  export BEAMFORMIT=$KALDI_ROOT/tools/BeamformIt
fi
export PATH=${PATH}:$BEAMFORMIT
! hash BeamformIt && echo "Missing BeamformIt, run 'cd ../../../tools/; make beamformit;'" && exit 1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

mkdir -p $odir
mkdir -p $wdir/log

# we use the following channel signals, and remove 2nd channel signal, which located on the back of
# tablet, and behaves very different from the other front channel signals.
bmf="1 3 4 5 6"
echo "Will use the following channels: $bmf"
# number of channels
numch=`echo $bmf | tr ' ' '\n' | wc -l`
echo "the number of channels: $numch"

# wavfiles.list can be used as the name of the output files
output_wavfiles=$wdir/wavfiles.list
find $sdir/*{simu,real} | grep CH1.wav \
  | awk -F '/' '{print $(NF-1) "/" $NF}' | sed -e "s/\.CH1\.wav//" | sort > $output_wavfiles

# this is an input file list of the microphones
# format: 1st_wav 2nd_wav ... nth_wav
input_arrays=$wdir/channels_$numch
for x in `cat $output_wavfiles`; do
  echo -n "$x"
  for ch in $bmf; do
    echo -n " $x.CH$ch.wav"
  done
  echo ""
done > $input_arrays

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
    --config_file `pwd`/conf/ami.cfg \
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
