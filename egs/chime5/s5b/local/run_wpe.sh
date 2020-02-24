#!/usr/bin/env bash
# Copyright 2018 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0

. ./cmd.sh
. ./path.sh

# Config:
nj=4
cmd=run.pl

. utils/parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 3)"
   echo "Usage: local/run_wpe.sh [options] <wav-in-dir> <wav-out-dir> <array-id>"
   echo "main options (for others, see top of script file)"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --nj 50                        # number of jobs for parallel processing"
   exit 1;
fi

sdir=$1
odir=$2
array=$3
task=`basename $sdir`
expdir=exp/wpe/${task}_${array}
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

mkdir -p $odir
mkdir -p $expdir/log

# wavfiles.list can be used as the name of the output files
output_wavfiles=$expdir/wavfiles.list
find -L ${sdir} | grep -i ${array} > $expdir/channels_input
cat $expdir/channels_input | awk -F '/' '{print $NF}' | sed "s@S@$odir\/S@g" > $expdir/channels_output
paste -d" " $expdir/channels_input $expdir/channels_output > $output_wavfiles

# split the list for parallel processing
split_wavfiles=""
for n in `seq $nj`; do
  split_wavfiles="$split_wavfiles $output_wavfiles.$n"
done
utils/split_scp.pl $output_wavfiles $split_wavfiles || exit 1;

echo -e "Dereverberation - $task - $array\n"
# making a shell script for each job
for n in `seq $nj`; do
cat <<-EOF > $expdir/log/wpe.$n.sh
while read line; do
  $miniconda_dir/bin/python local/run_wpe.py \
    --file \$line
done < $output_wavfiles.$n
EOF
done

chmod a+x $expdir/log/wpe.*.sh
$cmd JOB=1:$nj $expdir/log/wpe.JOB.log \
  $expdir/log/wpe.JOB.sh

echo "`basename $0` Done."
