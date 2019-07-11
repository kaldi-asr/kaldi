#!/bin/bash
#
# Copyright 2019  Johns Hopkins University (Author: Daniel Povey)
#           2019  Yiming Wang
# Apache 2.0


# Begin configuration section.
cmd=run.pl
stage=0
nj=4
wake_word="嗨小问"
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "usage: local/process_lattice.sh <lattice-dir> <data-dir> <lang-dir>"
  echo "e.g.:  local/process_lattice.sh --nj 100 exp/chain/tdnn_1a/decode_eval data/eval_hires data/lang"
  exit 1;
fi

dir=$1
data=$2
lang=$3

if [ $stage -le 1 ]; then
  mkdir -p $dir/scoring
  cat <<EOF >$dir/empty_word_fst.txt
0
EOF
  fstcompile $dir/empty_word_fst.txt $dir/empty_word.fst

  id=`cat $lang/words.txt | grep $wake_word | awk '{print $2}'`
  freetext_id=`cat $lang/words.txt | grep "FREETEXT" | awk '{print $2}'`
  sil_id=`cat $lang/words.txt | grep "<sil>" | awk '{print $2}'`
  cat <<EOF >$dir/wake_word_fst.txt
0 0 $sil_id $sil_id
0 0 $freetext_id $freetext_id
0 1 $id $id
1 1 $sil_id $sil_id
1 1 $freetext_id $freetext_id
1 1 $id $id
1
EOF
  fstcompile $dir/wake_word_fst.txt $dir/wake_word.fst
  $cmd JOB=1:$nj $dir/log/compute_cost_wake_word.JOB.log \
    lattice-to-fst --lm-scale=1.0 --acoustic-scale=1.0 "ark:gunzip -c $dir/lat.JOB.gz |" ark:- \| fsttablecomposelog $dir/wake_word.fst ark:- ark:- \| fsts-clear-labels ark:- ark:- \| fsttablecomposelog $dir/empty_word.fst ark:- ark:- \| fstdeterminizestar --use-log="true" ark:- ark,t:$dir/scoring/cost_wake_word.JOB.txt || exit 1;
  for n in $(seq $nj); do
    cat $dir/scoring/cost_wake_word.$n.txt || exit 1;
  done > $dir/scoring/cost_wake_word.txt || exit 1
  for n in $(seq $nj); do
    rm -f $dir/scoring/cost_wake_word.$n.txt 2>/dev/null || true
  done
  rm -f $dir/wake_word_fst.txt $dir/wake_word.fst 2>/dev/null || true

  cat <<EOF >$dir/non_wake_word_fst.txt
0 0 $sil_id $sil_id
0 0 $freetext_id $freetext_id
0
EOF
  fstcompile $dir/non_wake_word_fst.txt $dir/non_wake_word.fst
  $cmd JOB=1:$nj $dir/log/compute_cost_non_wake_word.JOB.log \
    lattice-to-fst --lm-scale=1.0 --acoustic-scale=1.0 "ark:gunzip -c $dir/lat.JOB.gz |" ark:- \| fsttablecomposelog $dir/non_wake_word.fst ark:- ark:- \| fsts-clear-labels ark:- ark:- \| fsttablecomposelog $dir/empty_word.fst ark:- ark:- \| fstdeterminizestar --use-log="true" ark:- ark,t:$dir/scoring/cost_non_wake_word.JOB.txt || exit 1;
  for n in $(seq $nj); do
    cat $dir/scoring/cost_non_wake_word.$n.txt || exit 1;
  done > $dir/scoring/cost_non_wake_word.txt || exit 1
  for n in $(seq $nj); do
    rm -f $dir/scoring/cost_non_wake_word.$n.txt 2>/dev/null || true
  done
  rm -f $dir/non_wake_word_fst.txt $dir/non_wake_word.fst 2>/dev/null || true
fi

if [ $stage -le 2 ]; then
  local/parse_cost.py $dir/scoring/cost_wake_word.txt $dir/scoring/cost_non_wake_word.txt > $dir/scoring/cost.txt
  dur=0
  [ -f $data/utt2dur ] && utils/filter_scp.pl <(grep -v $wake_word $data/text) $data/utt2dur > $data/utt2dur_negative && dur=`awk '{a+=$2} END{print a}' $data/utt2dur_negative`
  export LC_ALL=en_US.UTF-8
  paste -d' ' <(cat $dir/scoring/cost.txt) <(cut -f2 -d' ' $data/utt2dur) | awk '{a=($3-$2)/$4;print $1,a}' > $dir/scoring/score.txt
  python3 local/plot_scatter.py --wake-word $wake_word $dir/scoring/score.txt $data/text
  python3 local/compute_min_dcf.py --wake-word $wake_word --duration $dur $data/text $dir/scoring/score.txt
  export LC_ALL=C
fi
