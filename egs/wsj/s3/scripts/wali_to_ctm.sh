#!/bin/bash

# Produces ctm files suitable for scoring with NIST's sclite tool.
# Note: with 2 arguments this produces a ctm that's "relative to the utterance-id",
# i.e. it treats the utterance-id as if it's a file.
# If you provide the segments-file, 
# which specifies how the utterance-ids relate to the original waveform files,
# it produces output that is relative to the original waveform files.

if [ $# -ne 2 -a $# -ne 3 ]; then
  echo "Usage: wali_to_ctm.sh word-alignments words-symbol-table [segments-file] > ctm" 1>&2
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
         print "$utt 1 $time $dur $word\n"; 
       }
       $time =$time + $dur;
     } ' | scripts/int2sym.pl --field 5 $symtab | \
   ( if [ $# -eq 2 ]; then 
       cat
     else # Convert this ctm to being relative to orig.
           # waveform files.
       segments=$3
       [ ! -f $segments ] && echo No such file $segments && exit 1;
       perl -e '$seg=shift @ARGV; open(S, "<$seg")||die "No such file $seg";
         while(<S>){ ($utt,$spk,$begin,$end)=split;
           ($filename,$side) = split("-",$spk);  $begin{$utt}=$begin;
           $end{$utt}=$end;$filename{$utt}=$filename; $side{$utt}=$side;
         }           
         while(<STDIN>) {
           ($utt,$one,$time,$dur,$word)=split;
           $filename=$filename{$utt}; $side=$side{$utt};
           $begin=$begin{$utt}; 
           defined $begin && defined $filename && defined $side|| die "Bad utt $utt: not in segments file";
           $begintime = $time + $begin{$utt};
           print "$filename $side $begintime $dur $word\n";
         } ' $segments
       fi )
     
         


