#!/bin/bash

# Copyright 2014  Guoguo Chen
#           2017  Vimal Manohar
# Apache 2.0

# This script rescores non-compact, (possibly) undeterminized lattices with the 
# ConstArpaLm format language model.
# This is similar to steps/lmrescore_const_arpa.sh, but expects 
# non-compact lattices as input.
# This works by first determinizing the lattice and rescoring it with 
# const ARPA LM, followed by composing it with the original lattice to add the 
# new LM scores.

# If you use the option "--write compact false" it outputs non-compact lattices;
# the purpose is to add in LM scores while leaving the frame-by-frame acoustic
# scores in the same position that they were in in the input, undeterminized
# lattices. This is important in our 'chain' semi-supervised training recipes,
# where it helps us to split lattices while keeping the scores at the edges of
# the split points correct.

# Begin configuration section.
cmd=run.pl
skip_scoring=false
stage=1
scoring_opts=
write_compact=true   # If set to false, writes lattice in non-compact format.
                     # This retains the acoustic scores on the arcs of the lattice.
                     # Useful for another stage of LM rescoring.
acwt=0.1  # used for pruning and determinization
beam=8.0  # beam used in determinization

# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./utils/parse_options.sh

if [ $# != 5 ]; then
  cat <<EOF
   Does language model rescoring of non-compact undeterminized lattices 
   (remove old LM, add new LM). This script expects the input lattices 
   to be in non-compact format.
   Usage: $0 [options] <old-lang-dir> <new-lang-dir> \\
                      <data-dir> <input-decode-dir> <output-decode-dir>
   options: [--cmd (run.pl|queue.pl [queue opts])]
   See also: steps/lmrescore_const_arpa.sh 
EOF
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

if ! cmp -s $oldlang/words.txt $newlang/words.txt; then
  echo "$0: $oldlang/words.txt and $newlang/words.txt differ: make sure you know what you are doing.";
fi

oldlmcommand="fstproject --project_output=true $oldlm |"

mkdir -p $outdir/log
nj=`cat $indir/num_jobs` || exit 1;
cp $indir/num_jobs $outdir

lats_rspecifier="ark:gunzip -c $indir/lat.JOB.gz |"
  
lats_wspecifier="ark:| gzip -c > $outdir/lat.JOB.gz" 

if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
    lattice-determinize-pruned --acoustic-scale=$acwt --beam=$beam \
      "ark:gunzip -c $indir/lat.JOB.gz |" ark:- \| \
    lattice-scale --lm-scale=0.0 --acoustic-scale=0.0 ark:- ark:- \| \
    lattice-lmrescore --lm-scale=-1.0 ark:- "$oldlmcommand" ark:- \| \
    lattice-lmrescore-const-arpa --lm-scale=1.0 \
      ark:- "$newlm" ark:- \| \
    lattice-project ark:- ark:- \| \
    lattice-compose --write-compact=$write_compact \
      "$lats_rspecifier" \
      ark,s,cs:- "$lats_wspecifier" || exit 1
fi

if ! $skip_scoring && [ $stage -le 2 ]; then
  err_msg="Not scoring because local/score.sh does not exist or not executable."
  [ ! -x local/score.sh ] && echo $err_msg && exit 1;
  local/score.sh --cmd "$cmd" $scoring_opts $data $newlang $outdir
else
  echo "Not scoring because requested so..."
fi

exit 0;
