#!/bin/bash

cmd=scripts/run.pl
if [ $1 == "--cmd" ]; then
  shift
  cmd=$1
  shift
  [ -z "$cmd" ] && echo "Empty argument to --cmd option" && exit 1;
fi  

if [ $# != 6 ]; then
   echo "Do language model rescoring of lattices (remove old LM, add new LM)"
   echo "Usage: scripts/lmrescore.sh <old-LM-fst> <new-LM-fst> <words.txt> <data-dir> <input-decode-dir> <output-decode-dir>"
   exit 1;
fi

. path.sh || exit 1;



oldlm=$1
newlm=$2
words=$3
data=$4
indir=$5
outdir=$6

oldlmcommand="fstproject --project_output=true $oldlm |"
newlmcommand="fstproject --project_output=true $newlm |"

mkdir -p $outdir;
for lat in $indir/lat.*.gz; do
  number=`basename $lat | cut -d. -f2`;
  newlat=$outdir/`basename $lat`
  $cmd $outdir/rescorelm.$number.log \
    lattice-lmrescore --lm-scale=-1.0 "ark:gunzip -c $lat|" "$oldlmcommand" ark:-  \| \
    lattice-lmrescore --lm-scale=1.0 ark:- "$newlmcommand" "ark,t:|gzip -c>$newlat" &
done

wait
scripts/score_lats.sh $outdir $words $data

