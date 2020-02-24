#!/usr/bin/env bash

# Copyright 2015, Mitsubishi Electric Research Laboratories, MERL (Author: Shinji Watanabe)

. ./cmd.sh
. ./path.sh

# Config:
nj=10
cmd=run.pl
bmf="1 3 4 5 6"
eval_flag=true # make it true when the evaluation data are released

. utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Wrong #arguments ($#, expected 2)"
   echo "Usage: local/run_beamform_6ch_track.sh [options] <wav-in-dir> <wav-out-dir>"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --bmf \"1 3 4 5 6\"                      # microphones used for beamforming (2th mic is omitted in default)"
   exit 1;
fi

sdir=$1
odir=$2
wdir=data/beamforming_`echo $bmf | tr ' ' '_'`

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

echo "Will use the following channels: $bmf"
# number of channels
numch=`echo $bmf | tr ' ' '\n' | wc -l`
echo "the number of channels: $numch"

# wavfiles.list can be used as the name of the output files
# we will process train, dev, and eval waves
output_wavfiles=$wdir/wavfiles.list
if $eval_flag; then
  find $sdir/{tr,dt,et}*{simu,real}/ | grep CH1.wav \
    | awk -F '/' '{print $(NF-1) "/" $NF}' | sed -e "s/\.CH1\.wav//" | sort > $output_wavfiles
else
  find $sdir/{tr,dt}*{simu,real}/ | grep CH1.wav \
    | awk -F '/' '{print $(NF-1) "/" $NF}' | sed -e "s/\.CH1\.wav//" | sort > $output_wavfiles
fi

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
