#!/bin/bash

# Copyright 2016 Ricky Chan Ho Yin
# Apache 2.0

# This script rescores lattices with CUED RNNLM.  See also rnnlmrescore_cued.sh which is
# a script using n-best lists.

# Begin configuration section.
cmd=run.pl
skip_scoring=false
max_ngram_order=4
N=10
inv_acwt=13
weight=0.75  # Interpolation weight for RNNLM.
nthread=1
# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./utils/parse_options.sh

if [ $# != 5 ]; then
   echo "Does language model rescoring of lattices (remove old LM, add new LM)"
   echo "with RNNLM."
   echo ""
   echo "Usage: $0 [options] <old-lang-dir> <rnnlm-dir> \\"
   echo "                   <data-dir> <input-decode-dir> <output-decode-dir>"
   echo " e.g.: $0 ./rnnlm data/lang_tg data/test \\"
   echo "                   exp/tri3/test_tg exp/tri3/test_rnnlm"
   echo "options: [--cmd (run.pl|queue.pl [queue opts])]"
   exit 1;
fi

[ -f path.sh ] && . ./path.sh;

oldlang=$1
rnnlm_dir=$2
data=$3
indir=$4
outdir=$5

oldlm=$oldlang/G.fst
if [ -f $oldlang/G.carpa ]; then
  oldlm=$oldlang/G.carpa
elif [ ! -f $oldlm ]; then
  echo "$0: expecting either $oldlang/G.fst or $oldlang/G.carpa to exist" &&\
    exit 1;
fi

[ ! -f $oldlm ] && echo "$0: Missing file $oldlm" && exit 1;
[ ! -f $rnnlm_dir/rnnlm.txt ] && echo "$0: Missing file $rnnlm_dir/rnnlm.txt" && exit 1;
[ ! -f $rnnlm_dir/rnnlm.txt.input.wlist.index ] && echo "$0: Missing file $rnnlm_dir/rnnlm.txt.input.wlist.index" && exit 1;
[ ! -f $rnnlm_dir/rnnlm.txt.output.wlist.index ] && echo "$0: Missing file $rnnlm_dir/rnnlm.txt.output.wlist.index" && exit 1;
[ ! -f $rnnlm_dir/rnnlm.info ] && echo "$0: Missing file $rnnlm_dir/rnnlm.info" && exit 1;
[ ! -f $oldlang/words.txt ] &&\
  echo "$0: Missing file $oldlang/words.txt" && exit 1;
! ls $indir/lat.*.gz >/dev/null &&\
  echo "$0: No lattices input directory $indir" && exit 1;
awk -v n=$0 -v w=$weight 'BEGIN {if (w < 0 || w > 1) {
  print n": Interpolation weight should be in the range of [0, 1]"; exit 1;}}' \
  || exit 1;

oldlm_command="fstproject --project_output=true $oldlm |"

acwt=`perl -e "print (1.0/$inv_acwt);"`

mkdir -p $outdir/log
nj=`cat $indir/num_jobs` || exit 1;
cp $indir/num_jobs $outdir

oldlm_weight=`perl -e "print -1.0 * $weight;"`
if [ "$oldlm" == "$oldlang/G.fst" ]; then
  $cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
    lattice-lmrescore --lm-scale=$oldlm_weight \
    "ark:gunzip -c $indir/lat.JOB.gz|" "$oldlm_command" ark:-  \| \
    lattice-lmrescore-cuedrnnlm --lm-scale=$weight \
    --max-ngram-order=$max_ngram_order \
    --nthread=$nthread \
    $oldlang/words.txt ark:- "$rnnlm_dir/rnnlm.txt" \
    "ark,t:|gzip -c>$outdir/lat.JOB.gz" \
    "$rnnlm_dir/rnnlm.txt.input.wlist.index" "$rnnlm_dir/rnnlm.txt.output.wlist.index" "$rnnlm_dir/rnnlm.info" || exit 1;
else
  $cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
    lattice-lmrescore-const-arpa --lm-scale=$oldlm_weight \
    "ark:gunzip -c $indir/lat.JOB.gz|" "$oldlm_command" ark:-  \| \
    lattice-lmrescore-cuedrnnlm --lm-scale=$weight \
    --max-ngram-order=$max_ngram_order \
    $oldlang/words.txt ark:- "$rnnlm_dir/rnnlm.txt" \
    "ark,t:|gzip -c>$outdir/lat.JOB.gz" \
    "$rnnlm_dir/rnnlm.txt.input.wlist.index" "$rnnlm_dir/rnnlm.txt.output.wlist.index" "$rnnlm_dir/rnnlm.info" || exit 1;
fi

if ! $skip_scoring ; then
  err_msg="Not scoring because local/score.sh does not exist or not executable."
  [ ! -x local/score.sh ] && echo $err_msg && exit 1;
  local/score.sh --cmd "$cmd" $data $oldlang $outdir
else
  echo "Not scoring because requested so..."
fi

exit 0;
