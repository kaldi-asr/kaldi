#!/usr/bin/env bash
# Copyright 2018 Johns Hopkins University (Author: Aswin Shanmugan Subramamian)
#           2020 Bar Ben-Yair
#           2020 Desh Raj
# Apache 2.0

. ./cmd.sh
. ./path.sh

# Config:
cmd=run.pl

. utils/parse_options.sh || exit 1;

if [ $# != 1 ]; then
   echo "Wrong #arguments ($#, expected 1)"
   echo "Usage: local/run_wpe.sh [options] <wav-in-dir>"
   echo "main options (for others, see top of script file)"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --nj 50                        # number of jobs for parallel processing"
   exit 1;
fi

datadir=$1

expdir=exp/wpe/
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

miniconda_dir=$HOME/miniconda3/
if [ ! -d $miniconda_dir ]; then
    echo "$miniconda_dir does not exist. Please run '$KALDI_ROOT/tools/extras/install_miniconda.sh'."
    exit 1
fi

# check if WPE is installed
result=`$miniconda_dir/bin/python -c "\
try:
    import nara_wpe
    print('1')
except ImportError:
    print('0')"`

if [ "$result" == "1" ]; then
    echo "WPE is installed"
else
    echo "WPE is not installed. Please run ../../../tools/extras/install_wpe.sh"
    exit 1
fi

mkdir -p $datadir/wavs_dereverb/
mkdir -p $expdir/log

# We create a list of all wav files
ls ${datadir}/wavs/* > $expdir/channels_input

# split the list for parallel processing. We will create one job
# for each recording.
reco_ids=$( cat $expdir/channels_input | sed 's/_CH[0-7]//g' | sort -u )
nj=$( echo $reco_ids | wc -w )

for n in `seq $nj`; do
  # Find all channels for that recording and add to the wav list
  pattern=$( echo $reco_ids | tr ' ' '\n' | sed "${n}q;d" | awk -F'_' '{print $1"_CH._"$2}' )
  grep $pattern $expdir/channels_input > $expdir/split_wav.$n 
done

echo -e "Dereverberation using online WPE..\n"
# making a shell script for each job
for n in `seq $nj`; do
  cat <<-EOF > $expdir/log/wpe.$n.sh
  $miniconda_dir/bin/python local/run_wpe.py \
      $expdir/split_wav.$n $datadir/wavs_dereverb
EOF
done

chmod a+x $expdir/log/wpe.*.sh
$cmd JOB=1:$nj $expdir/log/wpe.JOB.log \
  $expdir/log/wpe.JOB.sh

rm -r $expdir

echo "`basename $0` Done."