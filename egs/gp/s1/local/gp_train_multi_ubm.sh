#!/bin/bash

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

nj=4      # Default number of jobs
qcmd=""   # Options for the submit_jobs.sh script
sjopts="" # Options for the submit_jobs.sh script

PROG=`basename $0`;
usage="Usage: $PROG [options] <num-comp> <out-dir>\n
e.g.: $PROG exp/ubm3a\n\n
Options:\n
  --help\t\tPrint this message and exit\n
  --num-jobs INT\tNumber of parallel jobs to run (default=$nj).\n
  --qcmd STR\tCommand for submitting a job to a grid engine (e.g. qsub) including switches.\n
  --sjopts STR\tOptions for the 'submit_jobs.sh' script\n
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
    *)   break ;;   # end of options: interpreted as number of components
  esac
done

if [ $# != 2 ]; then
  error_exit $usage;
fi

numcomps=$1
dir=$2

LANGUAGES='GE PO SP SW'  # Languages processed
[ -f path.sh ] && . ./path.sh
mkdir -p $dir/{data,log}
for f in feats.scp spk2utt utt2spk text wav.scp; do
  for L in $LANGUAGES; do
    cat data/$L/train/$f
  done \
    | sort -k1,1 > $dir/data/$f
done
data=$dir/data
split_data.sh $data $nj

# typically: --intermediate-numcomps=2000 --ubm-numcomps=400
intermediate=$[$numcomps*5]

merge_ubms=
for L in $LANGUAGES; do
  alidir=exp/$L/tri2a_ali
  merge_ubms=$merge_ubms" $dir/${L}.ubm"
  echo "Language '$L': Clustering model $alidir/final.mdl to get initial UBM"
  (
    submit_jobs.sh "$qcmd" --log=$dir/log/cluster_$L.log $sjopts \
      init-ubm --intermediate-numcomps=$intermediate --ubm-numcomps=$numcomps \
	--verbose=2 --fullcov-ubm=true $alidir/final.mdl $alidir/final.occs \
	$dir/${L}.ubm || touch $dir/.error
  ) &  # Run the language-specific clusterings in parallel
done
wait
[ -f $dir/.error ] && \
  { rm $dir/.error; error_exit "UBM initialization failed."; }

echo "Merging language-specific UBMs to a global UBM."
fgmm-global-merge $dir/0.ubm $dir/ubm_sizes $merge_ubms

echo "Computing cepstral mean and variance statistics"
submit_jobs.sh "$qcmd" --njobs=$nj $sjopts --log=$dir/log/cmvn.TASK_ID.log \
  compute-cmvn-stats --spk2utt=ark:$data/split$nj/TASK_ID/spk2utt \
    scp:$data/split$nj/TASK_ID/feats.scp ark:$dir/TASK_ID.cmvn \
    || error_exit "Computing CMN/CVN stats failed.";

feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/TASK_ID/utt2spk ark:$dir/TASK_ID.cmvn scp:$data/split$nj/TASK_ID/feats.scp ark:- | add-deltas ark:- ark:- |"

# First do Gaussian selection to 100 components, which will be used
# as the initial screen for all further passes.
ngselect=100
submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/log/gselect_diag.TASK_ID.log \
  $sjopts gmm-gselect --n=$ngselect "fgmm-global-to-gmm $dir/0.ubm - |" \
    "$feats" "ark:|gzip -c >$dir/gselect_diag.TASK_ID.gz" \
    || error_exit "Error doing GMM selection";
gs_opt="--gselect=ark,s,cs:gunzip -c $dir/gselect_diag.TASK_ID.gz|"

ngselect=50  # During iterations select 50 components
for x in 0 1 2 3; do
  echo "Pass $x"
  submit_jobs.sh "$qcmd" --njobs=$nj $sjopts --log=$dir/log/acc.$x.TASK_ID.log \
    gmm-gselect --n=$ngselect "$gs_opt" "fgmm-global-to-gmm $dir/$x.ubm - |" \
      "$feats" ark:- \| \
    fgmm-global-acc-stats --gselect=ark,s,cs:- $dir/$x.ubm "$feats" \
      $dir/$x.TASK_ID.acc \
    || error_exit "Error accumulating stats for UBM estimation on pass $x."

  # Only remove low-count Gaussians on last iter-- keeps gselect info valid.
  lowcount_opt="--remove-low-count-gaussians=false"
  [ $x -eq 3 ] && lowcount_opt=

  submit_jobs.sh "$qcmd" --log=$dir/log/update.$x.log $sjopts \
    fgmm-global-est $lowcount_opt --verbose=2 $dir/$x.ubm \
      "fgmm-global-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].ubm \
      || error_exit "Error estimating UBM on pass $x.";
  rm $dir/$x.*.acc $dir/$x.ubm
done

rm $dir/gselect_diag.*.gz
rm -f $dir/final.ubm
mv $dir/4.ubm $dir/final.ubm || exit 1;
