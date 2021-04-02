#!/usr/bin/env bash

# Copyright 2015, Mitsubishi Electric Research Laboratories, MERL (Author: Shinji Watanabe)

. ./cmd.sh
. ./path.sh

# Config:
cmd=run.pl
bmf="1 2 3 4"

. utils/parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 3)"
   echo "Usage: local/run_beamformit.sh [options] <wav-in-dir> <wav-out-dir> <array-id>"
   echo "main options (for others, see top of script file)"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --bmf \"1 2 3 4\"                        # microphones used for beamforming"
   exit 1;
fi

sdir=$1
odir=$2
array=$3
expdir=exp/enhan/`echo $odir | awk -F '/' '{print $NF}'`_`echo $bmf | tr ' ' '_'`

if ! command  -v BeamformIt &>/dev/null ; then
  echo "Missing BeamformIt, run 'cd $KALDI_ROOT/tools/; ./extras/install_beamformit.sh; cd -;'" && exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

mkdir -p $odir
mkdir -p $expdir/log

echo "Will use the following channels: $bmf"
# number of channels
numch=`echo $bmf | tr ' ' '\n' | wc -l`
echo "the number of channels: $numch"

# wavfiles.list can be used as the name of the output files
output_wavfiles=$expdir/wavfiles.list
find -L ${sdir} | grep -i ${array} | awk -F "/" '{print $NF}' | sed -e "s/\.CH.\.wav//" | sort | uniq > $expdir/wavfiles.list

# this is an input file list of the microphones
# format: 1st_wav 2nd_wav ... nth_wav
input_arrays=$expdir/channels_$numch
for x in `cat $output_wavfiles`; do
  echo -n "$x"
  for ch in $bmf; do
    echo -n " $x.CH$ch.wav"
  done
  echo ""
done > $input_arrays

# split the list for parallel processing
# number of jobs are set by the number of WAV files
nj=`wc -l $expdir/wavfiles.list | awk '{print $1}'`
split_wavfiles=""
for n in `seq $nj`; do
  split_wavfiles="$split_wavfiles $output_wavfiles.$n"
done
utils/split_scp.pl $output_wavfiles $split_wavfiles || exit 1;

echo -e "Beamforming\n"
# making a shell script for each job
for n in `seq $nj`; do
cat << EOF > $expdir/log/beamform.$n.sh
while read line; do
  $BEAMFORMIT/BeamformIt -s \$line -c $input_arrays \
    --config_file `pwd`/conf/beamformit.cfg \
    --source_dir $sdir \
    --result_dir $odir
done < $output_wavfiles.$n
EOF
done

chmod a+x $expdir/log/beamform.*.sh
$cmd JOB=1:$nj $expdir/log/beamform.JOB.log \
  $expdir/log/beamform.JOB.sh

echo "`basename $0` Done."
