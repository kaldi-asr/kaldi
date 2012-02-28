#!/bin/bash -u

# Copyright 2012  Arnab Ghoshal

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

# Converts HTK features to Kaldi format.

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

nj=1      # Default number of jobs
qcmd=""   # Options for the submit_jobs.sh script
sjopts="" # Options for the submit_jobs.sh script

PROG=`basename $0`;
usage="Usage: $PROG [options] <htk-file-list> <log-dir> <out-dir> <out-list>\n\n
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
      shift; qcmd="--qcmd=${1}"; shift ;;
    --sjopts)
      shift; sjopts="$1"; shift ;;
    -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
    *)   break ;;   # end of options: interpreted as the data-dir
  esac
done

if [ $# != 4 ]; then
  error_exit $usage;
fi

[ -f path.sh ] && . path.sh

htklist=$1
logdir=$2
outdir=$3
outlist=$4

# use "name" as part of name of the archive.
name=`basename $htklist`

mkdir -p $outdir || error_exit "Cannot create '$outdir'.";
mkdir -p $logdir || error_exit "Cannot create '$logdir'.";

# note: in general, the double-parenthesis construct in bash "((" is "C-style
# syntax" where we can get rid of the $ for variable names, and omit spaces.
# The "for" loop in this style is a special construct.

split_scps=""
for ((n=1; n<=nj; n++)); do
  split_scps="$split_scps $logdir/htk$n.scp"
done

split_scp.pl $htklist $split_scps || exit 1;

submit_jobs.sh "$qcmd" --njobs=$nj --log=$logdir/htk2kaldi.TASK_ID.log $sjopts \
  copy-feats --verbose=2 --htk-in scp:$logdir/htkTASK_ID.scp \
    ark,scp:$outdir/kaldi_$name.TASK_ID.ark,$outdir/kaldi_$name.TASK_ID.scp \
    || error_exit "Error converting HTK features:"`tail $logdir/htk2kaldi.*.log`

# concatenate the .scp files together.
rm -f $outlist
for ((n=1; n<=nj; n++)); do
  cat $outdir/kaldi_$name.$n.scp >> $outlist
done

rm $logdir/htk*.scp

echo "Succeeded in copying HTK featurs to Kaldi format."
