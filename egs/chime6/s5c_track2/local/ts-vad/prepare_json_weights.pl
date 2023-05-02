#!/usr/bin/perl
# Copyright     2020  Ivan Medennikov (STC-innovations Ltd)
# Apache 2.0.

# This script creates per-utterance json alignment by per-session alignment and segments.

($segments,$jsonali_scp,$jsonali_scp_perutt)=@ARGV;

%ark={};

open(fidin, "<$jsonali_scp") or die "can't open $jsonali_scp : $!";
while ($line=<fidin>)
{
  $line=~s/\s+$//;
  @items=split(/\s+/,$line);
  $ark{$items[0]}=$items[1];
  print "$items[0] $ark{$items[0]}\n";
}
close(fidin);

open(fidin, "<$segments") or die "can't open $segments : $!";
open(fidout, ">$jsonali_scp_perutt") or die "can't open $jsonali_scp_perutt : $!";
while ($line=<fidin>)
{
  $line=~s/\s+$//;
  @items=split(/\s+/,$line);
  $utt=$items[0];
  $wav=$items[1];
  $beg=$items[2];
  $end=$items[3];
  if ($utt=~/_(S\d+).*(\d{7})-(\d{7})/)
  {
    $sess=$1;
    $ubeg=$2;
    $ubeg=~s/^0+//;
    if ($utt=~/sp(\d+\.\d+)/)
    {
      $sp=$1;
      $ubeg=int($ubeg/$sp+0.5);
      $sess=$sess."_sp$sp";
    }
    if (($utt=~/^$wav\-\d+$/) || ($utt=~/^$wav$/))
    {
      $beg=$ubeg+int($beg*100+0.5);
      $end=$ubeg+int($end*100+0.5)-1;
    }
    else
    {
      $beg=int($beg*100+0.5);
      $end=int($end*100+0.5)-1;
    }
    print fidout "$utt $ark{$sess}\[$beg\:$end\]\n";
  }
}
close(fidin);
close(fidout);
exit 0;