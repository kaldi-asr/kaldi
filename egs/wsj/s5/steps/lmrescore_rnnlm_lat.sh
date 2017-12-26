#!/bin/bash

# Copyright 2015  Guoguo Chen
#           2017  Hainan Xu
# Apache 2.0

# This script rescores lattices with RNNLM.  See also rnnlmrescore.sh which is
# an older script using n-best lists.

# Begin configuration section.
cmd=run.pl
skip_scoring=false
max_ngram_order=4
acwt=0.1
weight=0.5  # Interpolation weight for RNNLM.
rnnlm_ver=
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

rescoring_binary=lattice-lmrescore-rnnlm

first_arg=ark:$rnnlm_dir/unk.probs # this is for mikolov's rnnlm
extra_arg=

if [ "$rnnlm_ver" == "cuedrnnlm" ]; then
  layer_string=`cat $rnnlm_dir/layer_string | sed "s=:= =g"`
  total_size=`wc -l $rnnlm_dir/unigram.counts | awk '{print $1}'`
  rescoring_binary="lattice-lmrescore-cuedrnnlm"
  cat $rnnlm_dir/rnnlm.input.wlist.index | tail -n +2 | awk '{print $1-1,$2}' > $rnnlm_dir/rnn.wlist
  extra_arg="--full-voc-size=$total_size --layer-sizes=\"$layer_string\""
  first_arg=$rnnlm_dir/rnn.wlist
fi

oldlm=$oldlang/G.fst
if [ -f $oldlang/G.carpa ]; then
  oldlm=$oldlang/G.carpa
elif [ ! -f $oldlm ]; then
  echo "$0: expecting either $oldlang/G.fst or $oldlang/G.carpa to exist" &&\
    exit 1;
fi

[ ! -f $oldlm ] && echo "$0: Missing file $oldlm" && exit 1;
[ ! -f $rnnlm_dir/rnnlm ] && echo "$0: Missing file $rnnlm_dir/rnnlm" && exit 1;
[ ! -f $rnnlm_dir/unk.probs ] &&\
  echo "$0: Missing file $rnnlm_dir/unk.probs" && exit 1;
[ ! -f $oldlang/words.txt ] &&\
  echo "$0: Missing file $oldlang/words.txt" && exit 1;
! ls $indir/lat.*.gz >/dev/null &&\
  echo "$0: No lattices input directory $indir" && exit 1;
awk -v n=$0 -v w=$weight 'BEGIN {if (w < 0 || w > 1) {
  print n": Interpolation weight should be in the range of [0, 1]"; exit 1;}}' \
  || exit 1;

oldlm_command="fstproject --project_output=true $oldlm |"

mkdir -p $outdir/log
nj=`cat $indir/num_jobs` || exit 1;
cp $indir/num_jobs $outdir

oldlm_weight=`perl -e "print -1.0 * $weight;"`
if [ "$oldlm" == "$oldlang/G.fst" ]; then
  $cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
    lattice-lmrescore --lm-scale=$oldlm_weight \
    "ark:gunzip -c $indir/lat.JOB.gz|" "$oldlm_command" ark:-  \| \
    $rescoring_binary $extra_arg --lm-scale=$weight \
    --max-ngram-order=$max_ngram_order \
    $first_arg $oldlang/words.txt ark:- "$rnnlm_dir/rnnlm" \
    "ark,t:|gzip -c>$outdir/lat.JOB.gz" || exit 1;
else
  $cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
    lattice-lmrescore-const-arpa --lm-scale=$oldlm_weight \
    "ark:gunzip -c $indir/lat.JOB.gz|" "$oldlm" ark:-  \| \
    $rescoring_binary $extra_arg --lm-scale=$weight \
    --max-ngram-order=$max_ngram_order \
    $first_arg $oldlang/words.txt ark:- "$rnnlm_dir/rnnlm" \
    "ark,t:|gzip -c>$outdir/lat.JOB.gz" || exit 1;
fi
if ! $skip_scoring ; then
  err_msg="Not scoring because local/score.sh does not exist or not executable."
  [ ! -x local/score.sh ] && echo $err_msg && exit 1;
  local/score.sh --cmd "$cmd" $data $oldlang $outdir
else
  echo "$0: Not scoring because --skip-scoring was specified."
fi

exit 0;
