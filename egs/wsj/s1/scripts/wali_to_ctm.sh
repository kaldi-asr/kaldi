#!/bin/bash


if [ $# != 2 ]; then
  echo "Usage: wali_to_ctm.sh word-alignments words-symbol-table > ctm" 1>&2
  exit 1;
fi


wali=$1
symtab=$2

cat $wali | \
  perl -ane '@A = split(" "); $utt = shift @A; @A = split(";", join(" ", @A));
     $time=0.0;
     foreach $a (@A) { 
       ($word,$dur) = split(" ", $a); 
       $dur *= 0.01;
       if ($word != 0) {
         print "$utt 1 $word $time $dur $word\n"; 
       }
       $time =$time + $dur;
     } ' | scripts/int2sym.pl --field 6 $symtab



