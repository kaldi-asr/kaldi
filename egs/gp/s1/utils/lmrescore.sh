#!/usr/bin/env bash

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
mode=4
qcmd=""   # Options for the submit_jobs.sh script
sjopts="" # Options for the submit_jobs.sh script

PROG=`basename $0`;
usage="Usage: $PROG [options] <old-lang-dir> <new-lang-dir> <data-dir> <input-decode-dir> <output-decode-dir>\n
Do language model rescoring of lattices (remove old LM, add new LM)\n\n
Options:\n
  --help\t\tPrint this message and exit\n
  --mode INT\tOptions for the decoder script\n
  --qcmd STRING\tCommand for submitting a job to a grid engine (e.g. qsub) including switches.\n
  --sjopts STRING\tOptions for the 'submit_jobs.sh' script\n
";

while [ $# -gt 0 ]; do
  case "${1# *}" in  # ${1# *} strips any leading spaces from the arguments
    --help) echo -e $usage; exit 0 ;;
    --mode)
      shift; mode=`readint $1`;
      [ $mode -lt 1 ] && error_exit "--mode arg '$mode' not positive.";
      shift ;;
    --qcmd)
      shift; qcmd="--qcmd=${1}"; shift ;;
    --sjopts)
      shift; sjopts="$1"; shift ;;
    -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
    *)   break ;;   # end of options: interpreted as the old LM directory
  esac
done

if [ $# != 5 ]; then
  error_exit $usage
fi

[ -f path.sh ] && . ./path.sh

oldlang=$1
newlang=$2
data=$3
indir=$4
outdir=$5

oldlm=$oldlang/G.fst
newlm=$newlang/G.fst
! cmp $oldlang/words.txt $newlang/words.txt && echo "Warning: vocabularies may be incompatible."
[ ! -f $oldlm ] && echo Missing file $oldlm && exit 1;
[ ! -f $newlm ] && echo Missing file $newlm && exit 1;
! ls $indir/lat.*.gz >/dev/null && echo "No lattices input directory $indir" && exit 1;


oldlmcommand="fstproject --project_output=true $oldlm |"
newlmcommand="fstproject --project_output=true $newlm |"

mkdir -p $outdir;

phi=`grep -w '#0' $newlang/words.txt | awk '{print $2}'`

if [ "$mode" == 4 ]; then
  # we have to prepare $outdir/Ldet.fst in this case: determinized
  # lexicon, with disambig syms removed.
  grep '#' $newlang/phones_disambig.txt | awk '{print $2}' > $outdir/disambig.list
  # take L_disambig.fst; get rid of transition with "#0 #0" on it; determinize
  # with epsilon removal; remove disambiguation symbols.
  fstprint $newlang/L_disambig.fst | awk '{if($4 != '$phi'){print;}}' | fstcompile | \
    fstdeterminizestar | fstrmsymbols $outdir/disambig.list >$outdir/Ldet.fst || exit 1;
fi

# for lat in $indir/lat.*.gz; do
nj=`ls $indir/lat.*.gz | wc -l`  # Number of lattices found
echo $nj;
for n in `seq 1 $nj`; do  # Make sure lattices are indexed properly
  [ -f $indir/lat.$n.gz ] || error_exit "Lattice '$indir/lat.$n.gz' not found."
done
exit

lat=$indir/lat.TASK_ID.gz
newlat=$outdir/`basename $lat`
case "$mode" in
  1) # 1 is inexact, the original way of doing it.
    submit_jobs.sh "$qcmd" --njobs=$nj --log=$outdir/rescorelm.TASK_ID.log \
      $sjopts lattice-lmrescore --lm-scale=-1.0 "ark:gunzip -c $lat|" \
      "$oldlmcommand" ark:- \| lattice-lmrescore --lm-scale=1.0 ark:- \
      "$newlmcommand" "ark,t:|gzip -c>$newlat" \
      || error_exit "Error doing LM rescoring."
    ;;
   2)  # 2 is equivalent to 1, but using more basic operations, combined.
    submit_jobs.sh "$qcmd" --njobs=$nj --log=$outdir/rescorelm.TASK_ID.log \
      $sjopts gunzip -c $lat \| \
      lattice-scale --acoustic-scale=-1 --lm-scale=-1 ark:- ark:- \| \
      lattice-compose ark:- "fstproject --project_output=true $oldlm |" ark:- \
      \| lattice-determinize ark:- ark:- \| \
      lattice-scale --acoustic-scale=-1 --lm-scale=-1 ark:- ark:- \| \
      lattice-compose ark:- "fstproject --project_output=true $newlm |" ark:- \
      \| lattice-determinize ark:- ark:- \| \
      gzip -c \>$newlat || error_exit "Error doing LM rescoring."
    ;;
  3) # 3 is "exact" in that we remove the old LM scores accepting any path
     # through G.fst (which is what we want as that happened in lattice 
     # generation), but we add the new one with "phi matcher", only taking
     # backoff arcs if an explicit arc did not exist.
    submit_jobs.sh "$qcmd" --njobs=$nj --log=$outdir/rescorelm.TASK_ID.log \
      $sjopts gunzip -c $lat \| \
      lattice-scale --acoustic-scale=-1 --lm-scale=-1 ark:- ark:- \| \
      lattice-compose ark:- "fstproject --project_output=true $oldlm |" ark:- \
      \| \ lattice-determinize ark:- ark:- \| \
      lattice-scale --acoustic-scale=-1 --lm-scale=-1 ark:- ark:- \| \
      lattice-compose --phi-label=$phi ark:- $newlm ark:- \| \
      lattice-determinize ark:- ark:- \| \
      gzip -c \>$newlat || error_exit "Error doing LM rescoring."
    ;;
  4) # 4 is also exact (like 3), but instead of subtracting the old LM-scores,
     # it removes the old graph scores entirely and adds in the lexicon,
     # grammar and transition weights.
    mdl=`dirname $indir`/final.mdl
    [ ! -f $mdl ] && echo No such model $mdl && exit 1;
    submit_jobs.sh "$qcmd" --njobs=$nj --log=$outdir/rescorelm.TASK_ID.log \
      $sjopts gunzip -c $lat \| \
      lattice-scale --lm-scale=0.0 ark:- ark:- \| \
      lattice-to-phone-lattice $mdl ark:- ark:- \| \
      lattice-compose ark:- $outdir/Ldet.fst ark:- \| \
      lattice-determinize ark:- ark:- \| \
      lattice-compose --phi-label=$phi ark:- $newlm ark:- \| \
      lattice-add-trans-probs --transition-scale=1.0 --self-loop-scale=0.1 \
        $mdl ark:- ark:- \| \
      gzip -c \>$newlat  ||  error_exit "Error doing LM rescoring."
  ;;
esac

rm $outdir/Ldet.fst 2>/dev/null
score_lats.sh $outdir $newlang/words.txt $data

