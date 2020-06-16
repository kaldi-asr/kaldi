#!/usr/bin/perl
# Copyright     2020  Ivan Medennikov (STC-innovations Ltd)
# Apache 2.0.

($filein,$utt2spk,$utt2dur,$chunk,$fileout,$fileout2,$fileout3)=@ARGV;

%utt2dur={};
open(fidin, "<$utt2dur") or die "can't open $utt2dur : $!";
while ($line=<fidin>)
{
  $line=~s/\s+$//; 
  @items=split(/\s+/,$line);
  $utt2dur{$items[0]}=$items[1];
}
close(fidin);

%utt2spk={};
open(fidin, "<$utt2spk") or die "can't open $utt2spk : $!";
while ($line=<fidin>)
{
  $line=~s/\s+$//;
  @items=split(/\s+/,$line);
  $utt2spk{$items[0]}=$items[1];
}
close(fidin);


open(fidin, "<$filein") or die "can't open $filein : $!";
open(fidout, ">$fileout") or die "can't open $fileout : $!";
open(fidout2, ">$fileout2") or die "can't open $fileout2 : $!";
open(fidout3, ">$fileout3") or die "can't open $fileout3 : $!";

while ($line=<fidin>)
{
  $line=~s/\s+$//;
  @items=split(/\s+/,$line);
  $begin=0;
  $end=$begin+$chunk-1;
  $id=1;
  $suffix=$id;
  while ($begin < $utt2dur{$items[0]})
  {
    $end=$begin+$chunk-1;
    if ($end > $utt2dur{$items[0]}-1)
    {
        $end = $utt2dur{$items[0]}-1;
    }
    if ($id < 1000)
    {
        $suffix="0$id";
    }
    if ($id < 100)
    {
        $suffix="00$id";
    }
    if ($id < 10)
    {
        $suffix="000$id";
    }
    print fidout "$items[0]-$suffix $items[1]\[$begin:$end\]\n";
    print fidout2 "$items[0]-$suffix $utt2spk{$items[0]}\n";
    $begin_sec=$begin/100.0;
    $end_sec=$end/100.0;
    print fidout3 "$items[0]-$suffix $items[0] $begin_sec $end_sec\n";
    $begin=$begin+$chunk;
    $id=$id+1;
  }
}
close(fidin);
close(fidout);
close(fidout2);
close(fidout3);
exit 0;