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

# Decoding script for SGMM using standard MFCC/PLP + delta + acceleration 
# features.

# assumes you are using speaker vectors [for no vectors, see 
# decode_sgmm_novec_lda_etc.sh, if it exists already].
# if this includes speaker-specific transforms, you have to provide an "old" 
# decoding directory where the transforms are located.  The data decoded in 
# that directory must be split up in the same way as the current directory.

function error_exit () {
  echo -e "$@" >&2; exit 1;
}

function file_exists () {
  [ -f $1 ] || error_exit "$PROG: no such file '$1'"
}

function readposint () {  # Strictly speaking, reading non-negative integers
  local retval=${1/#*=/};  # In case --switch=ARG format was used
  [[ "$retval" =~ ^[0-9]*$ ]] \
    || error_exit "Argument \"$retval\" not a non-negative integer."
  echo $retval
}

beam=13.0
nj=1       # Default total number of jobs
jobid=0    # Default job number
qcmd=""    # Options for the submit_jobs.sh script
sjopts=""  # Options for the submit_jobs.sh script
use_spkvecs=''  # Not expecting a model with speaker vectors, by default.

PROG=`basename $0`;
usage="Usage: $PROG [options] <graph-dir> <data-dir> <decode-dir> [<transform-dir>]\n
e.g.: $PROG -j 10 0 exp/sgmm3c/graph_tgpr data/test_dev93 exp/sgmm3c/decode_dev93_tgpr exp/tri2b/decode_dev93_tgpr\n\n
Options:\n
  --help\t\tPrint this message and exit.\n
  --beam FLOAT\tDecoding beam (default=$beam).\n
  -j INT INT\tNumber of parallel jobs to run (default=$nj) and current jobid.\n
  --qcmd STRING\tCommand for submitting a job to a grid engine (e.g. qsub) including switches.\n
  --sjopts STRING\tOptions for the 'submit_jobs.sh' script.\n
  --with-spkvecs\tModel has speaker vectors; do 2-pass decoding.\n
";

while [ $# -gt 0 ]; do
  case "${1# *}" in  # ${1# *} strips any leading spaces from the arguments
    --help) echo -e $usage; exit 0 ;;
    --beam) beam=$2; shift 2 ;;
    -j) nj=`readposint $2`; jobid=`readposint $3`; shift 3 ;;
    --qcmd) qcmd=" --qcmd=${2}"; shift 2 ;;
    --sjopts) sjopts="$2"; shift 2 ;;
    --with-spkvecs) use_spkvecs=1; shift ;;
    -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
    *)   break ;;   # end of options: interpreted as num-leaves
  esac
done

if [ $# -lt 3 -o $# -gt 4 ]; then
  error_exit $usage;
fi

[ -f path.sh ] && . path.sh

graphdir=$1
data=$2
dir=$3
transdir=$4
acwt=0.1  # Just a default value, used for adaptation and beam-pruning..

srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

mkdir -p $dir

if [ $nj -gt 1 ]; then
  mydata=$data/split$nj/$jobid
else
  mydata=$data
fi

requirements="$mydata/feats.scp $srcdir/final.mdl $graphdir/HCLG.fst"
[ -z "$use_spkvecs" ] || requirements=$requirements" $srcdir/final.alimdl"
for f in $requirements; do
  file_exists $f
done

if [ ! -z "$transdir" ]; then # "$transdir" nonempty..
  file_exists $transdir/$n.trans
fi

feats="ark:compute-cmvn-stats --spk2utt=ark:$mydata/spk2utt scp:$mydata/feats.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$mydata/utt2spk ark:- scp:$mydata/feats.scp ark:- | add-deltas ark:- ark:- |"

[ ! -z "$transdir" ] && feats="$feats transform-feats --utt2spk=ark:$mydata/utt2spk ark:$transdir/$jobid.trans ark:- ark:- |"


# Do Gaussian selection, since we'll have two decoding passes and don't want to 
# redo this. Note: it doesn't make a difference if we use final.mdl or 
# final.alimdl, they have the same UBM.
sgmm-gselect $srcdir/final.mdl "$feats" "ark:|gzip -c >$dir/$jobid.gselect.gz" \
  2>$dir/gselect$jobid.log \
  || error_exit "Error in Gaussian selection.";
gselect_opt="--gselect=ark:gunzip -c $dir/$jobid.gselect.gz|"

target_lat="$dir/lat.$jobid.gz"
[ -z "$use_spkvecs" ] || target_lat="$dir/pre_lat.$jobid.gz"
align_model="$srcdir/final.mdl"
[ -z "$use_spkvecs" ] || align_model="$srcdir/final.alimdl"

# Generate a state-level lattice for rescoring, with the alignment model and no 
# speaker vectors.

sgmm-latgen-faster --max-active=7000 --beam=$beam --lattice-beam=6.0 \
  --acoustic-scale=$acwt --determinize-lattice=false --allow-partial=true \
  --word-symbol-table=$graphdir/words.txt "$gselect_opt" $align_model \
  $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $target_lat" \
  2> $dir/decode_pass1.$jobid.log \
  || error_exit "Error in 1st-pass decoding.";

# Do a second pass "decoding" if using speaker vectors.
if [ ! -z "$use_spkvecs" ]; then
  silphonelist=`cat $graphdir/silphones.csl` || exit 1
  ( lattice-determinize --acoustic-scale=$acwt --prune=true --beam=4.0 \
      "ark:gunzip -c $dir/pre_lat.$jobid.gz|" ark:- \
    | lattice-to-post --acoustic-scale=$acwt ark:- ark:- \
    | weight-silence-post 0.0 $silphonelist $srcdir/final.alimdl ark:- ark:- \
    | sgmm-post-to-gpost "$gselect_opt" $srcdir/final.alimdl "$feats" ark:- \
      ark:- \
    | sgmm-est-spkvecs-gpost --spk2utt=ark:$mydata/spk2utt $srcdir/final.mdl \
      "$feats" ark:- "ark:$dir/$jobid.vecs" 
  ) 2> $dir/vecs.$jobid.log \
    || error_exit "Error estimating speaker vectors.";

  # Now rescore the state-level lattices with the adapted features and the
  # corresponding model. Prune and determinize the lattices to limit their size.

  sgmm-rescore-lattice "$gselect_opt" --utt2spk=ark:$mydata/utt2spk \
    --spk-vecs=ark:$dir/$jobid.vecs $srcdir/final.mdl \
    "ark:gunzip -c $dir/pre_lat.$jobid.gz|" "$feats" \
    "ark:|lattice-determinize --acoustic-scale=$acwt --prune=true --beam=6.0 ark:- ark:- | gzip -c > $dir/lat.$jobid.gz" \
    2>$dir/rescore.$jobid.log \
    || error_exit "Error in 2nd-pass rescoring.";

  rm $dir/pre_lat.$jobid.gz
  # The top-level decoding script rescores "lat.$jobid.gz" to get final output.
fi

