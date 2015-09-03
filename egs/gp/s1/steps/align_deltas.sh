#!/bin/bash

# Copyright 2010-2012  Microsoft Corporation;  Arnab Ghoshal

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

# This script does training-data alignment given a model built using 
# CMN + delta + delta-delta features.  It splits the data into
# four chunks and does everything in parallel on the same machine.
# Its output, all in its own experimental directory, is (assuming
# you don't change the #jobs with --num-job option),
# {0,1,2,3}.cmvn {0,1,2,3}.ali.gz, tree, final.mdl 
# and final.occs (the last three are just copied from the source directory). 


# Option to use precompiled graphs from last phase, if these
# are available (i.e. if they were built with the same data).
# These must be split into four pieces.

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
oldgraphs=false

PROG=`basename $0`;
usage="Usage: $PROG [options]  <data-dir> <lang-dir> <src-dir> <exp-dir>\n
e.g.: $PROG data/train data/lang exp/tri1 exp/tri1_ali\n\n
Options:\n
  --help\t\tPrint this message and exit\n
  --num-jobs INT\tNumber of parallel jobs to run (default=$nj).\n
  --qcmd STRING\tCommand for submitting a job to a grid engine (e.g. qsub) including switches.\n
  --sjopts STRING\tOptions for the 'submit_jobs.sh' script\n
  --use-graphs\tReuse older graphs\n
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
    --use-graphs)
      oldgraphs=true; shift ;;
    -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
    *)   break ;;   # end of options: interpreted as the data-dir
  esac
done

if [ $# != 4 ]; then
  error_exit $usage;
fi

[ -f path.sh ] && . path.sh

data=$1
lang=$2
srcdir=$3
dir=$4

oov_sym=`cat $lang/oov.txt`

mkdir -p $dir
# Create copy of the tree and model and occs...
cp $srcdir/{tree,final.mdl,final.occs} $dir || exit 1;

scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"

if [ ! -d $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  split_data.sh $data $nj
fi

echo "Computing cepstral mean and variance statistics"
# for n in `get_splits.pl $nj`; do # Do this locally; it's fast.
submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/cmvnTASK_ID.log $sjopts \
  compute-cmvn-stats --spk2utt=ark:$data/split$nj/TASK_ID/spk2utt \
    scp:$data/split$nj/TASK_ID/feats.scp ark:$dir/TASK_ID.cmvn \
    || error_exit "Computing CMN/CVN stats failed.";


# Align all training data using the supplied model.
echo "Aligning data from $data"
feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$dir/TASK_ID.cmvn scp:$data/split$nj/TASK_ID/feats.scp ark:- | add-deltas ark:- ark:- |"

if $oldgraphs; then 
  # for n in `get_splits.pl $nj`; do
  # feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$dir/TASK_ID.cmvn scp:$data/split$nj/TASK_ID/feats.scp ark:- | add-deltas ark:- ark:- |"
  ls $srcdir/{1..$nj}.fsts.gz >/dev/null \
    || error_exit "Missing FSTs with --use-graphs option specified."
  submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/alignTASK_ID.log $sjopts \
    gmm-align-compiled $scale_opts --beam=10 --retry-beam=40 $dir/final.mdl \
      "ark:gunzip -c $srcdir/TASK_ID.fsts.gz|" "$feats" "ark:|gzip -c >$dir/TASK_ID.ali.gz" \
      || error_exit "Error doing alignment.";

else
  # for n in `get_splits.pl $nj`; do
  # feats="ark:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:$dir/TASK_ID.cmvn scp:$data/split$nj/TASK_ID/feats.scp ark:- | add-deltas ark:- ark:- |"
  # compute integer form of transcripts.
  tra="ark:sym2int.pl --map-oov '$oov_sym' --ignore-first-field $lang/words.txt $data/split$nj/TASK_ID/text|";
  # We could just use gmm-align in the next line, but it's less efficient as 
  # it compiles the training graphs one by one.
  submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/alignTASK_ID.log $sjopts \
    compile-train-graphs $dir/tree $dir/final.mdl  $lang/L.fst "$tra" ark:- \| \
      gmm-align-compiled $scale_opts --beam=10 --retry-beam=40 $dir/final.mdl \
      ark:- "$feats" "ark:|gzip -c >$dir/TASK_ID.ali.gz" \
      || error_exit "Error doing alignment.";
fi

echo "Done aligning data."
