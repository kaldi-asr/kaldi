#!/bin/bash -u

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

# Decoding script that works with a GMM model and delta-delta plus
# cepstral mean subtraction features.  Used, for example, to decode
# mono/ and tri1/
# This script just generates lattices for a single broken-up
# piece of the data.

function error_exit () {
  echo -e "$@" >&2; exit 1;
}

function readfloat () {
  local retval=${1/#*=/};  # In case --switch=ARG format was used
  [[ "$retval" =~ ^-?[0-9]*\.*[0-9]*$ ]] \
    || error_exit "Argument \"$retval\" not a real number."
  echo $retval
}

function readint () {
  local retval=${1/#*=/};  # In case --switch=ARG format was used
  retval=${retval#0*}      # Strip any leading 0's
  [[ "$retval" =~ ^-?[1-9][0-9]*$ ]] \
    || error_exit "Argument \"$retval\" not an integer."
  echo $retval
}

accwt=1.0
beam=30.0
latgen=0
njobs=4
qcmd=""   # Options for the submit_jobs.sh script

PROG=`basename $0`;
usage="Usage: $PROG [options] <graph-dir> <data-dir> <decode-dir>\n
e.g.: $PROG exp/mono/graph_bg data/dev exp/mono/decode_dev_bg\n\n
Options:\n
  --help\t\tPrint this message and exit\n
  --accwt FLOAT\tScaling for acoustic likelihoods (default=$accwt).\n
  --beam FLOAT\tDecoder beam (default=$beam)\n
  --latgen\tGenerate lattices (off by default)\n
  --num-jobs INT\tNumber of parallel jobs to run (default=$njobs).\n
  --qcmd STRING\tCommand for submitting a job to a grid engine (e.g. qsub) including switches.\n
";

while [ $# -gt 0 ]; do
  case "${1# *}" in  # ${1# *} strips any leading spaces from the arguments
    --help) echo -e $usage; exit 0 ;;
    --accwt)
      shift; accwt=`readfloat $1`; shift ;;
    --beam)
      shift; beam=`readfloat $1`; shift ;;
    --latgen) shift; latgen=1 ;;
    --num-jobs)
      shift; njobs=`readint $1`;
      [ $njobs -lt 1 ] && error_exit "--num-jobs arg '$njobs' not positive.";
      shift ;;
    --qcmd)
      shift; qcmd="--qcmd=${1}"; shift ;;
    -*)  error_exit "Unknown argument: $1, exiting\n$usage" ;;
    *)   break ;;   # end of options: interpreted as the data-dir
  esac
done

if [ $# != 3 ]; then
  error_exit $usage;
fi

[ -f path.sh ] && . path.sh

graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

mkdir -p $dir

requirements="$data/feats.scp $srcdir/final.mdl $graphdir/HCLG.fst"
for f in $requirements; do
  if [ ! -f $f ]; then
    echo "decode_deltas.sh: no such file $f";
    exit 1;
  fi
done

# We only do one decoding pass, so there is no point caching the
# CMVN stats-- we make them part of a pipe.
feats="ark:compute-cmvn-stats --spk2utt=ark:$data/spk2utt scp:$data/feats.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk ark:- scp:$data/feats.scp ark:- | add-deltas ark:- ark:- |"
if [ $njobs -gt 1 ]; then
  if [ ! -d $data/split$njobs -o $data/split$njobs -ot $data/feats.scp ]; then
    split_data.sh $data $njobs
  fi
  mydata=$data/split$njobs/TASK_ID
  feats="ark:compute-cmvn-stats --spk2utt=ark:$mydata/spk2utt scp:$mydata/feats.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$mydata/utt2spk ark:- scp:$mydata/feats.scp ark:- | add-deltas ark:- ark:- |"
fi

if [ $latgen -eq 1 ]; then
  submit_jobs.sh "$qcmd" --njobs=$njobs --log=$dir/decode.TASK_ID.log \
    gmm-latgen-faster --max-active=7000 --beam=$beam --lattice-beam=6.0 \
      --acoustic-scale=$accwt --word-symbol-table=$graphdir/words.txt \
      $srcdir/final.mdl $graphdir/HCLG.fst "$feats" \
      "ark:|gzip -c > $dir/lat.TASK_ID.gz" || error_exit "Decoding failed.";
else
  submit_jobs.sh "$qcmd" --njobs=$njobs --log=$dir/decode.TASK_ID.log \
    gmm-decode-faster --beam=$beam --acoustic-scale=$accwt \
      --word-symbol-table=$graphdir/words.txt $srcdir/final.mdl \
      $graphdir/HCLG.fst "$feats" ark,t:$dir/test.TASK_ID.tra \
      || error_exit "Decoding failed.";
fi
