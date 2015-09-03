#!/bin/bash

# Copyright 2014  Guoguo Chen
# Apache 2.0

# This script rescores lattices with the ConstArpaLm format language model.

# Begin configuration section.
cmd=run.pl
skip_scoring=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./utils/parse_options.sh

if [ $# != 5 ]; then
   echo "Does language model rescoring of lattices (remove old LM, add new LM)"
   echo "Usage: steps/lmrescore.sh [options] <old-lang-dir> <new-lang-dir> \\"
   echo "                   <data-dir> <input-decode-dir> <output-decode-dir>"
   echo "options: [--cmd (run.pl|queue.pl [queue opts])]"
   exit 1;
fi

[ -f path.sh ] && . ./path.sh;

oldlang=$1
newlang=$2
data=$3
indir=$4
outdir=$5

oldlm=$oldlang/G.fst
newlm=$newlang/G.carpa
! cmp $oldlang/words.txt $newlang/words.txt &&\
  echo "$0: Warning: vocabularies may be incompatible."
[ ! -f $oldlm ] && echo "$0: Missing file $oldlm" && exit 1;
[ ! -f $newlm ] && echo "$0: Missing file $newlm" && exit 1;
! ls $indir/lat.*.gz >/dev/null &&\
  echo "$0: No lattices input directory $indir" && exit 1;

oldlmcommand="fstproject --project_output=true $oldlm |"

mkdir -p $outdir/log
nj=`cat $indir/num_jobs` || exit 1;
cp $indir/num_jobs $outdir

$cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
  lattice-lmrescore --lm-scale=-1.0 \
  "ark:gunzip -c $indir/lat.JOB.gz|" "$oldlmcommand" ark:-  \| \
  lattice-lmrescore-const-arpa --lm-scale=1.0 \
  ark:- "$newlm" "ark,t:|gzip -c>$outdir/lat.JOB.gz" || exit 1;

if ! $skip_scoring ; then
  err_msg="Not scoring because local/score.sh does not exist or not executable."
  [ ! -x local/score.sh ] && echo $err_msg && exit 1;
  local/score.sh --cmd "$cmd" $data $newlang $outdir
else
  echo "Not scoring because requested so..."
fi

exit 0;
