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

# To be run from ..
# Flat start and monophone training, with delta-delta features.
# This script applies cepstral mean normalization (per speaker).

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

nj=4       # Default number of jobs
stage=-4   # Default starting stage (start with calculating CMN/CVN stats)
qcmd=""    # Options for the submit_jobs.sh script
sjopts=""  # Options for the submit_jobs.sh script

PROG=`basename $0`;
usage="Usage: $PROG [options] <data-dir> <lang-dir> <exp-dir>\n
e.g.: $PROG data/train.1k data/lang exp/mono\n\n
Options:\n
  --help\t\tPrint this message and exit\n
  --num-jobs INT\tNumber of parallel jobs to run (default=$nj).\n
  --qcmd STRING\tCommand for submitting a job to a grid engine (e.g. qsub) including switches.\n
  --stage INT\tStarting stage (e.g. -4 for CMN/CVN stats; 2 for iter 2; default=$stage)\n
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
      shift; qcmd="--qcmd=${1}"; shift ;;
    --sjopts)
      shift; sjopts="$1"; shift ;;
    --stage)
      shift; stage=`readint $1`; shift ;;
    -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
    *)   break ;;   # end of options: interpreted as the data-dir
  esac
done

if [ $# != 3 ]; then
  error_exit $usage;
fi

data=$1
lang=$2
dir=$3

[ -f path.sh ] && . ./path.sh

# Configuration:
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
numiters=40    # Number of iterations of training
maxiterinc=30 # Last iter to increase #Gauss on.
numgauss=300 # Initial num-Gauss (must be more than #states=3*phones).
totgauss=1000 # Target #Gaussians.
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss
realign_iters="1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 23 26 29 32 35 38";
oov_sym=`cat $lang/oov.txt`

mkdir -p $dir/log
if [ ! -d $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  split_data.sh $data $nj
fi

if [ $stage -le -3 ]; then
  echo "Computing cepstral mean and variance statistics"
  # for n in `get_splits.pl $nj`; do # do this locally; it's fast.
  submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/log/cmvnTASK_ID.log $sjopts \
    compute-cmvn-stats --spk2utt=ark:$data/split$nj/TASK_ID/spk2utt \
      scp:$data/split$nj/TASK_ID/feats.scp ark:$dir/TASK_ID.cmvn \
      || error_exit "Computing CMN/CVN stats failed.";
fi

feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk \"ark:cat $dir/*.cmvn|\" scp:$data/feats.scp ark:- | add-deltas ark:- ark:- |"

# for n in `get_splits.pl $nj`; do
# for n in `seq 1 $nj`; do
featspart="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/TASK_ID/utt2spk ark:$dir/TASK_ID.cmvn scp:$data/split$nj/TASK_ID/feats.scp ark:- | add-deltas ark:- ark:- |"


if [ $stage -le -2 ]; then
  echo "Initializing monophone system."
  if [ -f $lang/phonesets_mono.txt ]; then
    echo "Using shared phones from $lang/phonesets_mono.txt"
    # In recipes with stress and position markers, this pools together
    # the stats for the different versions of the same phone (also for
    # the various silence phones).
    sym2int.pl $lang/phones.txt $lang/phonesets_mono.txt > $dir/phonesets.int
    shared_phones_opt="--shared-phones=$dir/phonesets.int"
  fi

  gmm-init-mono $shared_phones_opt \
    "--train-feats=$feats subset-feats --n=10 ark:- ark:-|" $lang/topo 39  \
    $dir/0.mdl $dir/tree 2> $dir/log/init.log \
    || error_exit "Monophone model initialization failed.";
fi

if [ $stage -le -1 ]; then
  echo "Compiling training graphs"
  submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/log/compile_graphsTASK_ID.log \
    $sjopts compile-train-graphs $dir/tree $dir/0.mdl $lang/L.fst \
      "ark:sym2int.pl --map-oov '$oov_sym' --ignore-first-field $lang/words.txt < $data/split$nj/TASK_ID/text|" \
      "ark:|gzip -c >$dir/TASK_ID.fsts.gz" \
      || error_exit "Error compiling training graphs.";
fi

if [ $stage -le 0 ]; then
  echo "Aligning data equally (pass 0)"
# for n in `get_splits.pl $nj`; do
  submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/log/align.0.TASK_ID.log \
    $sjopts align-equal-compiled "ark:gunzip -c $dir/TASK_ID.fsts.gz|" \
      "$featspart" ark,t,f:- \| \
    gmm-acc-stats-ali --binary=true $dir/0.mdl "$featspart" \
      ark:- $dir/0.TASK_ID.acc \
    || error_exit "Error in pass 0 accumulation";

# In the following steps, the --min-gaussian-occupancy=3 option is important,
# otherwise we cannot est "rare" phones and later on, they never align properly.
  gmm-est --min-gaussian-occupancy=3 --mix-up=$numgauss \
    $dir/0.mdl "gmm-sum-accs - $dir/0.*.acc|" $dir/1.mdl \
    2> $dir/log/update.0.log || error_exit "Error in pass 0 estimation.";

  rm $dir/0.*.acc
fi  # Finished 0'th training iteration.

beam=6  # will change to 10 below after 1st pass
x=1
while [ $x -lt $numiters ]; do
  echo "Pass $x"
  if [ $stage -le $x ]; then
    if echo $realign_iters | grep -w $x >/dev/null; then
      echo "Aligning data"
      submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/log/align.$x.TASK_ID.log \
        $sjopts gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$[$beam*4] \
        $dir/$x.mdl "ark:gunzip -c $dir/TASK_ID.fsts.gz|" "$featspart" \
        "ark,t:|gzip -c >$dir/TASK_ID.ali.gz" \
        || error_exit "Error in pass $x alignment.";
    fi  # Realign iters

    # for n in `get_splits.pl $nj`; do
    submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/log/acc.$x.TASK_ID.log \
      $sjopts gmm-acc-stats-ali $dir/$x.mdl "$featspart" \
      "ark:gunzip -c $dir/TASK_ID.ali.gz|" $dir/$x.TASK_ID.acc \
      || error_exit "Error in pass $x accumulation.";

    submit_jobs.sh "$qcmd" --log=$dir/log/update.$x.log $sjopts \
      gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc|" $dir/$[$x+1].mdl \
      || error_exit "Error in pass $x extimation.";
    rm -f $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs
  fi  # Completed a training stage.

  if [ $x -le $maxiterinc ]; then
    numgauss=$[$numgauss+$incgauss];
  fi
  beam=10
  x=$[$x+1];
done

( cd $dir; rm -f final.{mdl,occs}; ln -s $x.mdl final.mdl; \
  ln -s $x.occs final.occs; )

# Print out summary of the warning messages.
for x in $dir/log/*.log; do
  n=`grep WARNING $x | wc -l`;
  if [ $n -ne 0 ]; then echo $n warnings in $x; fi;
done

echo Done

# example of showing the alignments:
# show-alignments data/lang/phones.txt $dir/30.mdl "ark:gunzip -c $dir/0.ali.gz|" | head -4

