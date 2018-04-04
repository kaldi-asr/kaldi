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

function read_dirname () {
  local dir_name=${1/#*=/};  # In case --switch=ARG format was used
  [ -d "$dir_name" ] || error_exit "Argument '$dir_name' not a directory";
  local retval=`cd $dir_name 2>/dev/null && pwd || exit 1`
  echo $retval
}

orig_args="$*"
nj=""  # Total number of jobs unset by default. Will set to #speakers (if 
          # using a grid) or 4 (if not), unless specified by user.
lang=""   # Option for sclite scoring (off by default)
opts=""
qcmd=""   # Options for the submit_jobs.sh script
sjopts="" # Options for the submit_jobs.sh script

PROG=`basename $0`;
usage="Usage: $PROG [options] <decode_script> <graph-dir> <data-dir> <decode-dir> [extra-args...]\n\n
Options:\n
  --help\t\tPrint this message and exit\n
  -l DIR\t\tDirectory to find L_align.fst (needed for sclite scoring)\n
  --num-jobs INT\tNumber of parallel jobs to run (default=$nj).\n
  --opts STRING\tOptions for the decoder script\n
  --qcmd STRING\tCommand for submitting a job to a grid engine (e.g. qsub) including switches.\n
  --sjopts STRING\tOptions for the 'submit_jobs.sh' script\n
";

while [ $# -gt 0 ]; do
  case "${1# *}" in  # ${1# *} strips any leading spaces from the arguments
    --help) echo -e $usage; exit 0 ;;
    -l) 
      shift; lang=`read_dirname $1`;
      [ ! -f "$lang/phones_disambig.txt" -o ! -f "$lang/L_align.fst" ] && \
	error_exit "Invalid argument to -l option; expected $lang/phones_disambig.txt and $lang/L_align.fst to exist."
      shift ;;
    --num-jobs)
      shift; nj=`readint $1`;
      [ $nj -lt 1 ] && error_exit "--num-jobs arg '$nj' not positive.";
      shift ;;
    --opts)
      shift; opts="$1"; shift ;;
    --qcmd)
      shift; qcmd="--qcmd=${1}"; shift ;;
    --sjopts)
      shift; sjopts="$1"; shift ;;
    -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
    *)   break ;;   # end of options: interpreted as the script to execute
  esac
done


if [ $# -lt 4 ]; then
  error_exit $usage;
fi

script=$1
graphdir=$2
data=$3
dir=$4
# Make "dir" an absolute pathname.
dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`
mkdir -p $dir || exit 1
shift;shift;shift;shift;
# Remaining args will be supplied to decoding script.
extra_args=$* 

[ -f path.sh ] && . ./path.sh

for file in $script $scp $data/utt2spk; do
  if [ ! -f "$file" ]; then
     echo "decode.sh: no such file $file"
     exit 1
  fi 
done

if [ ! -f $graphdir/HCLG.fst -a ! -f $graphdir/G.fst ]; then
  # Note: most scripts expect HCLG.fst in graphdir, but the
  # "*_fromlats.sh" script(s) require(s) a "lang" dir in that
  # position
  echo No such file: $graphdir/HCLG.fst or $graphdir/G.fst
  exit 1;
fi

if [ -z "$nj" ]; then # Figure out num-jobs; user did not specify.
  if [ -z "$qcmd" ]; then
    nj=4
  else  # running on queue...
    nj=`utt2spk_to_spk2utt.pl $data/utt2spk | wc -l`
  fi
fi

echo "Decoding with num-jobs = $nj"
if [[ $nj -gt 1 || ! -d $data/split$nj || \
      $data/split$nj -ot $data/feats.scp ]]; then
  split_data.sh $data $nj
fi

#for n in `get_splits.pl $nj`; do
submit_jobs.sh "$qcmd" --njobs=$nj --log=$dir/decodeTASK_ID.log $sjopts \
  $script $opts -j $nj TASK_ID $graphdir $data $dir $extra_args \
  || error_exit "Error in decoding script: command was decode.sh $orig_args"

if ls $dir/lat.*.gz >&/dev/null; then
  if [ -n "$lang" ]; then  
  # sclite scoring: $lang directory supplied only for this reason.
    [ ! -f $data/stm ] && \
      error_exit "Expected $data/stm to exist (-l only used for sclite scoring)"
    score_lats_ctm.sh $dir $lang $data || \
      error_exit "Error in scoring of lattices using sclite."
  else
    score_lats.sh $dir $graphdir/words.txt $data || \
      error_exit "Error in scoring of latices.";
  fi
elif ls $dir/*.txt >&/dev/null; then
  score_text.sh $dir $data || error_exit "Error in scoring of hypotheses.";
else
  eror_exit "No output found in $dir, not scoring.";
fi
