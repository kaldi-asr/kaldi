#!/bin/bash


mode=4
cmd=scripts/run.pl
for x in 1 2; do
  if [ "$1" == "--cmd" ]; then
    shift
    cmd=$1
    shift
    [ -z "$cmd" ] && echo "Empty argument to --cmd option" && exit 1;
  fi  
  if [ "$1" == "--mode" ]; then
    shift
    mode=$1
    shift
  fi
done

if [ $# != 5 ]; then
   echo "Do language model rescoring of lattices (remove old LM, add new LM)"
   echo "Usage: scripts/lmrescore.sh <old-lang-dir> <new-lang-dir> <data-dir> <input-decode-dir> <output-decode-dir>"
   exit 1;
fi

. path.sh || exit 1;


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
[ ! ls $indir/lat.*.gz >/dev/null ] && echo "No lattices input directory $indir" && exit 1;


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


rm $outdir/.error 2>/dev/null

for lat in $indir/lat.*.gz; do
  number=`basename $lat | cut -d. -f2`;
  newlat=$outdir/`basename $lat`
  case "$mode" in
    1) # 1 is inexact, the original way of doing it.
  $cmd $outdir/rescorelm.$number.log \
    lattice-lmrescore --lm-scale=-1.0 "ark:gunzip -c $lat|" "$oldlmcommand" ark:-  \| \
    lattice-lmrescore --lm-scale=1.0 ark:- "$newlmcommand" "ark,t:|gzip -c>$newlat" \
        || touch $outdir/.error &
   ;;
   2)  # 2 is equivalent to 1, but using more basic operations, combined.
  $cmd $outdir/rescorelm.$number.log \
    gunzip -c $lat \| \
    lattice-scale --acoustic-scale=-1 --lm-scale=-1 ark:- ark:- \| \
    lattice-compose ark:- "fstproject --project_output=true $oldlm |" ark:- \| \
    lattice-determinize ark:- ark:- \| \
    lattice-scale --acoustic-scale=-1 --lm-scale=-1 ark:- ark:- \| \
    lattice-compose ark:- "fstproject --project_output=true $newlm |" ark:- \| \
    lattice-determinize ark:- ark:- \| \
    gzip -c \>$newlat || touch $outdir/.error &
    ;;
  3) # 3 is "exact" in that we remove the old LM scores accepting any path
     # through G.fst (which is what we want as that happened in lattice 
     # generation), but we add the new one with "phi matcher", only taking
     # backoff arcs if an explicit arc did not exist.
  $cmd $outdir/rescorelm.$number.log \
    gunzip -c $lat \| \
    lattice-scale --acoustic-scale=-1 --lm-scale=-1 ark:- ark:- \| \
    lattice-compose ark:- "fstproject --project_output=true $oldlm |" ark:- \| \
    lattice-determinize ark:- ark:- \| \
    lattice-scale --acoustic-scale=-1 --lm-scale=-1 ark:- ark:- \| \
    lattice-compose --phi-label=$phi ark:- $newlm ark:- \| \
    lattice-determinize ark:- ark:- \| \
    gzip -c \>$newlat || touch $outdir/.error &
  ;;
  4) # 4 is also exact (like 3), but instead of subtracting the old LM-scores,
     # it removes the old graph scores entirely and adds in the lexicon,
     # grammar and transition weights.
  mdl=`dirname $indir`/final.mdl
  [ ! -f $mdl ] && echo No such model $mdl && exit 1;
  $cmd $outdir/rescorelm.$number.log \
   gunzip -c $lat \| \
   lattice-scale --lm-scale=0.0 ark:- ark:- \| \
   lattice-to-phone-lattice $mdl ark:- ark:- \| \
   lattice-compose ark:- $outdir/Ldet.fst ark:- \| \
   lattice-determinize ark:- ark:- \| \
   lattice-compose --phi-label=$phi ark:- $newlm ark:- \| \
   lattice-add-trans-probs --transition-scale=1.0 --self-loop-scale=0.1 \
         $mdl ark:- ark:- \| \
     gzip -c \>$newlat  || touch $outdir/.error &
  ;;
  esac
done

wait
[ -f $outdir/.error ] && echo Error doing LM rescoring && exit 1
rm $outdir/Ldet.fst 2>/dev/null
scripts/score_lats.sh $outdir $newlang/words.txt $data

