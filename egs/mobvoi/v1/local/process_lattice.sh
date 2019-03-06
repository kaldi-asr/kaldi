#!/bin/bash
#
# Copyright 2019  Johns Hopkins University (Author: Daniel Povey)
#           2019  Yiming Wang
# Apache 2.0

# process lattices (lat.*.gz) to plot DET curves

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
  $cmd JOB=1:$nj $dir/log/copy_lattice.JOB.log \
    lattice-copy "ark:gunzip -c $dir/lat.JOB.gz |" ark,t:$dir/lat.JOB.txt || exit 1;
fi

if [ $stage -le 2 ]; then
  for n in $(seq $nj); do
    cat $dir/lat.$n.txt || exit 1;
  done > $dir/lat.txt || exit 1
  for n in $(seq $nj); do
    rm -f $dir/lat.$n.txt 2>/dev/null || true
  done
fi

if [ $stage -le 3 ]; then
  mkdir -p $dir/scoring
  cat <<EOF >$dir/empty_word_fst.txt
0
EOF
  fstcompile $dir/empty_word_fst.txt $dir/empty_word.fst

  id=`cat $lang/words.txt | grep $wake_word | awk '{print $2}'`
  freetext_id=`cat $lang/words.txt | grep "FREETEXT" | awk '{print $2}'`
  sil_id=`cat $lang/words.txt | grep "<sil>" | awk '{print $2}'`
  cat <<EOF >$dir/wake_word_fst.txt
0 1 $sil_id $sil_id
1 2 $id $id
0 2 $id $id
2 3 $sil_id $sil_id
2
3
EOF
  fstcompile $dir/wake_word_fst.txt $dir/wake_word.fst
  $cmd JOB=1:$nj $dir/log/compute_cost_wake_word.JOB.log \
    lattice-to-fst --lm-scale=1.0 --acoustic-scale=1.0 "ark:gunzip -c $dir/lat.JOB.gz |" ark:- \| fsttablecompose $dir/wake_word.fst ark:- ark:- \| fsts-clear-labels ark:- ark:- \| fsttablecomposelog $dir/empty_word.fst ark:- ark:- \| fstdeterminizestar --use-log="true" ark:- ark,t:$dir/scoring/cost_wake_word.JOB.txt || exit 1;
  for n in $(seq $nj); do
    cat $dir/scoring/cost_wake_word.$n.txt || exit 1;
  done > $dir/scoring/cost_wake_word.txt || exit 1
  for n in $(seq $nj); do
    rm -f $dir/scoring/cost_wake_word.$n.txt 2>/dev/null || true
  done
  rm -f $dir/wake_word_fst.txt $dir/wake_word.fst 2>/dev/null || true

  cat <<EOF >$dir/non_wake_word_fst.txt
0 1 $sil_id $sil_id
1 2 $freetext_id $freetext_id
0 2 $freetext_id $freetext_id
2 3 $sil_id $sil_id
2
3
EOF
#0 1 $sil_id $sil_id
#1
  fstcompile $dir/non_wake_word_fst.txt $dir/non_wake_word.fst
  $cmd JOB=1:$nj $dir/log/compute_cost_non_wake_word.JOB.log \
    lattice-to-fst --lm-scale=1.0 --acoustic-scale=1.0 "ark:gunzip -c $dir/lat.JOB.gz |" ark:- \| fsttablecompose $dir/non_wake_word.fst ark:- ark:- \| fsts-clear-labels ark:- ark:- \| fsttablecomposelog $dir/empty_word.fst ark:- ark:- \| fstdeterminizestar --use-log="true" ark:- ark,t:$dir/scoring/cost_non_wake_word.JOB.txt || exit 1;
  for n in $(seq $nj); do
    cat $dir/scoring/cost_non_wake_word.$n.txt || exit 1;
  done > $dir/scoring/cost_non_wake_word.txt || exit 1
  for n in $(seq $nj); do
    rm -f $dir/scoring/cost_non_wake_word.$n.txt 2>/dev/null || true
  done
  rm -f $dir/non_wake_word_fst.txt $dir/non_wake_word.fst 2>/dev/null || true
fi

if [ $stage -le 4 ]; then
  local/parse_cost.py $dir/scoring/cost_wake_word.txt $dir/scoring/cost_non_wake_word.txt > $dir/scoring/cost.txt
  dur=0
  [ -f $data/utt2dur ] && dur=`awk '{a+=$2} END{print a}' $data/utt2dur`
  export LC_ALL=en_US.UTF-8
  local/detect_from_cost.py --thres 0.0 --wake-word $wake_word $dir/scoring/cost.txt > $dir/scoring/detection.txt
  #cat $dir/scoring/cost.txt | awk '{if($2 == 0.0) print $1, ""; else if($3==0.0) print $1, "嗨小问";}' > $dir/scoring/detection.txt
  local/compute_metrics.py --wake-word $wake_word --duration $dur $data/text $dir/scoring/detection.txt 2>/dev/null | tee $dir/scoring/results
  cat $dir/scoring/cost.txt | awk '{a=$3-$2;print $1,a}' > $dir/scoring/score.txt
  /home/ywang/anaconda3/bin/python local/compute_min_dcf.py --wake-word $wake_word --duration $dur $data/text $dir/scoring/score.txt
  export LC_ALL=
fi
