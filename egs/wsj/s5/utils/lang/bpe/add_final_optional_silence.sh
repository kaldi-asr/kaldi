#!/bin/bash
. ./path.sh

final_sil_prob=0.5

echo "$0 $@"  # Print the command line for logging

. ./utils/parse_options.sh

if [ $# -ne 1 ]; then
  echo "Usage: $0  <lang>"
  echo " Add final optional silence to lexicon FSTs (L.fst and L_disambig.fst) in"
  echo " lang/ directory <lang>."
  echo " This can be useful in systems with byte-pair encoded (BPE) lexicons, in which"
  echo " the word-initial silence is part of the lexicon, so we turn off the standard"
  echo " optional silence in the lexicon"
  echo "options:"
  echo "   --final-sil-prob <final silence probability>      # default 0.5"
  exit 1;
fi

lang=$1

if [ $lang/phones/final_sil_prob -nt $lang/phones/nonsilence.txt ]; then
  echo "$0 $lang/phones/final_sil_prob exists. Exiting..."
  exit 1;
fi

silphone=$(cat $lang/phones/optional_silence.int)

sil_eq_zero=$(echo $(perl -e "if ( $final_sil_prob == 0.0) {print 'true';} else {print 'false';}"))
sil_eq_one=$(echo $(perl -e "if ( $final_sil_prob == 1.0) {print 'true';} else {print 'false';}"))
sil_lt_zero=$(echo $(perl -e "if ( $final_sil_prob < 0.0) {print 'true';} else {print 'false';}"))
sil_gt_one=$(echo $(perl -e "if ( $final_sil_prob > 1.0) {print 'true';} else {print 'false';}"))

if  $sil_lt_zero || $sil_gt_one; then
  echo "$0 final-sil-prob should be between 0.0 and 1.0. Final silence was not added."
  exit 1;
else
  if $sil_eq_zero; then
    echo "$0 final-sil-prob = 0 => Final silence was not added."
    exit 0;
  elif $sil_eq_one; then
    ( echo "0 1 $silphone 0";
      echo "1" ) | fstcompile > $lang/final_sil.fst
  else
    log_silprob=$(echo $(perl -e "print log $final_sil_prob"))
    ( echo "0 1 $silphone 0 $log_silprob";
      echo "0 $log_silprob";
      echo "1" ) | fstcompile > $lang/final_sil.fst
  fi
  mv $lang/L.fst $lang/L.fst.orig
  mv $lang/L_disambig.fst $lang/L_disambig.fst.orig
  fstconcat $lang/L.fst.orig $lang/final_sil.fst | fstarcsort --sort_type=olabel > $lang/L.fst
  fstconcat $lang/L_disambig.fst.orig $lang/final_sil.fst | fstarcsort --sort_type=olabel > $lang/L_disambig.fst
  echo "$final_sil_prob" > $lang/phones/final_sil_prob
fi
