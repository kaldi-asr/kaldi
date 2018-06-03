#!/bin/bash
. ./path.sh

final_sil_prob=0.5

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh

if [ $# -ne 1 ]; then
  echo "Usage: $0  <lang>"
  echo " Add final optional silence to lexicon FSTs (L.fst and L_disambig.fst) in  "
  echo " lang/ directory <lang>."
  echo "options:"
  echo "   --final-sil-prob <final silence probability>		# default 0.5"
  exit 1;
fi

langdir=$1

sil_zero=$(echo $(perl -e "if ( $final_sil_prob == 0.0) {print 1;} else {print 0;}"))
sil_one=$(echo $(perl -e "if ( $final_sil_prob == 1.0) {print 1;} else {print 0;}"))
sil_ge_zero=$(echo $(perl -e "if ( $final_sil_prob >= 0.0) {print 1;} else {print 0;}"))
sil_le_one=$(echo $(perl -e "if ( $final_sil_prob <= 1.0) {print 1;} else {print 0;}"))
if [ $sil_ge_zero -eq 1 ] && [ $sil_le_one -eq 1 ]
then
  if [ $sil_zero -eq 1 ]
  then
    echo "$0 final-sil-prob = 0 => Final silence was not added."
  elif [ $sil_one -eq 1 ]
  then
    log=$(echo $(perl -e "print log $final_sil_prob"))
    mv $langdir/L.fst $langdir/L.fst.orig
    mv $langdir/L_disambig.fst $langdir/L_disambig.fst.orig
    echo -e "0\t1\t1\t0\n1" | fstcompile > $langdir/final_sil.fst
    fstconcat $langdir/L.fst.orig $langdir/final_sil.fst | fstarcsort --sort_type=olabel > $langdir/L.fst
    fstconcat $langdir/L_disambig.fst.orig $langdir/final_sil.fst | fstarcsort --sort_type=olabel > $langdir/L_disambig.fst
  else
    log=$(echo $(perl -e "print log $final_sil_prob"))
    mv $langdir/L.fst $langdir/L.fst.orig
    mv $langdir/L_disambig.fst $langdir/L_disambig.fst.orig
    echo -e "0\t1\t1\t0\t$log\n0\t$log\n1\t0.0" | fstcompile > $langdir/final_sil.fst
    fstconcat $langdir/L.fst.orig $langdir/final_sil.fst | fstarcsort --sort_type=olabel > $langdir/L.fst
    fstconcat $langdir/L_disambig.fst.orig $langdir/final_sil.fst | fstarcsort --sort_type=olabel > $langdir/L_disambig.fst
  fi
else
  echo "$0 final-sil-prob should be between 0.0 and 1.0. Final silence was not added."
  exit 1;
fi

