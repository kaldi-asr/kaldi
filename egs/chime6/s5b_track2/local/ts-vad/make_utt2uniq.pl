#!/usr/bin/perl
# Copyright     2020  Ivan Medennikov (STC-innovations Ltd)
# Apache 2.0.

# This script creates utt2uniq file for the CHiME-6 utterances.

($filein,$fileout)=@ARGV;

open(fidout, ">$fileout") or die "can't open $fileout : $!";
open(fidin, "<$filein") or die "can't open $filein : $!";
while ($line=<fidin>)
{
  $line=~s/\s+$//;
  @items=split(/\s+/,$line);
  $utt=$items[0];
  $spk=$items[1];
  if ($utt=~/P(\d+)/)
  {
    $P=$1;
  }
  else
  {
    print "skipping utt $utt\n";
    next;
  }
  if ($utt=~/S(\d+)/)
  {
    $S=$1;
  }
  else
  {
    print "skipping utt $utt\n";
    next;
  }
  if ($utt=~/\D(\d{7})-(\d{7})/)
  {
    $beg=$1;
    $end=$2;
  }
  else
  {
    print "skipping utt $utt\n";
    next;
  }
  $id="P$P\_S$S\_$beg\-$end";
  print fidout "$utt $id\n";
}
close(fidin);
close(fidout);

exit 0;