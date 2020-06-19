#!/usr/bin/env bash

# Copyright 2012  Arnab Ghoshal
# Copyright 2010-2011  Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Train UBM from a trained HMM/GMM system using (e.g. MFCC) + delta + 
# acceleration features and cepstral mean normalization. 
# Alignment directory is used for the CMN and transforms.
# A UBM is just a single mixture of Gaussians (full-covariance, in our case), 
# that's trained on all the data.  This will later be used in SGMM training.

function error_exit () {
  echo -e "$@" >&2; exit 1;
}

function readint () {
  local retval=${1/#*=/};  # In case --switch=ARG format was used
  retval=${retval#0*}      # Strip any leading 0's
  [[ "$retval" =~ ^-?[1-9][0-9]*$ ]] \
    || error_exit "Argument \"$retval\" not an integer."
  echo $retval
}

nj=4   # Default number of jobs
qcmd=""   # Options for the submit_jobs.sh script
sjopts="" # Options for the submit_jobs.sh script

PROG=`basename $0`;
usage="Usage: $PROG [options] <num-comps> <data-dir> <ali-dir> <exp-dir>\n
e.g.: $PROG 400 data/train_si84 exp/tri2a_ali exp/ubm2a\n\n
Options:\n
  --help\t\tPrint this message and exit\n
  --num-jobs INT\tNumber of parallel jobs to run (default=$nj).\n
  --qcmd STRING\tCommand for submitting a job to a grid engine (e.g. qsub) including switches.\n
  --sjopts STRING\tOptions for the 'submit_jobs.sh' script\n
";

while [ $# -gt 0 ]; do
  case "${1# *}" in  # ${1# *} strips any leading spaces from the arguments
    --help) echo -e $usage; exit 0 ;;
    --num-jobs) 
      shift; nj=`readint $1`;
      [ $nj -lt 1 ] && error_exit "--num-jobs arg '$nj' not positive.";
      shift ;;
    --qcmd)
      shift; qcmd=" --qcmd=${1}"; shift ;;
    --sjopts)
      shift; sjopts="$1"; shift ;;
    -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
    *)   break ;;   # end of options: interpreted as num-leaves
  esac
done

if [ $# != 4 ]; then
  error_exit $usage;
fi

[ -f path.sh ] && . ./path.sh

numcomps=$1
data=$2
alidir=$3
dir=$4

mkdir -p $dir/log

if [ ! -d $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  scripts/split_data.sh $data $nj
fi

featspart="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/TASK_ID/utt2spk ark:$alidir/TASK_ID.cmvn scp:$data/split$nj/TASK_ID/feats.scp ark:- | add-deltas ark:- ark:- |"

ngselect1=50
ngselect2=25

intermediate=2000
if [ $[$numcomps*2] -gt $intermediate ]; then
  intermediate=$[$numcomps*2];
fi

echo "Clustering model $alidir/final.mdl to get initial UBM"
# typically: --intermediate-numcomps=2000 --ubm-numcomps=400

submit_jobs.sh "$qcmd" --log=$dir/log/cluster.log $sjopts \
  init-ubm --intermediate-numcomps=$intermediate --ubm-numcomps=$numcomps \
   --verbose=2 --fullcov-ubm=true $alidir/final.mdl $alidir/final.occs \
    $dir/0.ubm || error_exit "UBM initialization failed.";

# First do Gaussian selection to 50 components, which will be used
# as the initial screen for all further passes.
# for n in `get_splits.pl $nj`; do
submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/log/gselect_diag.TASK_ID.log \
  $sjopts gmm-gselect --n=$ngselect1 "fgmm-global-to-gmm $dir/0.ubm - |" \
    "$featspart" "ark:|gzip -c >$dir/gselect_diag.TASK_ID.gz" \
    || error_exit "Error doing GMM selection";

for x in 0 1 2 3; do
  echo "Pass $x"
#  for n in `get_splits.pl $nj`; do
  submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/log/acc.$x.TASK_ID.log \
    $sjopts gmm-gselect --n=$ngselect2 "--gselect=ark,s,cs:gunzip -c $dir/gselect_diag.TASK_ID.gz|" \
      "fgmm-global-to-gmm $dir/$x.ubm - |" "$featspart" ark:- \| \
      fgmm-global-acc-stats --gselect=ark,s,cs:- $dir/$x.ubm "$featspart" \
      $dir/$x.TASK_ID.acc \
      || error_exit "Error accumulating stats for UBM estimation on pass $x."
  lowcount_opt="--remove-low-count-gaussians=false"
  [ $x -eq 3 ] && lowcount_opt=   # Only remove low-count Gaussians on last iter-- keeps gselect info valid.
  submit_jobs.sh "$qcmd" --log=$dir/log/update.$x.log $sjopts \
    fgmm-global-est $lowcount_opt --verbose=2 $dir/$x.ubm \
      "fgmm-global-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].ubm \
      || error_exit "Error estimating UBM on pass $x.";
  rm $dir/$x.*.acc $dir/$x.ubm
done

rm $dir/gselect_diag.*.gz
rm $dir/final.ubm 2>/dev/null
mv $dir/4.ubm $dir/final.ubm || exit 1;
