#!/usr/bin/perl

($filein,$ups,$fileout)=@ARGV;

open(fidin, "<$filein") or die "cant open $filein : $!";
open(fidout, ">$fileout") or die "cant open $fileout : $!";
%utt2spk={};
%spk2utt={};
while ($line=<fidin>)
{
  $line=~s/\s+$//;
  @items=split(/\s+/,$line);
  $utt=$items[0];
  $spk=$items[1];
  push (@{$spk2utt{$spk}},$utt);
}
close(fidin);

foreach $spk (sort keys %{spk2utt})
{
  $i=0;
  $num=scalar @{$spk2utt{$spk}};
  foreach $utt (sort @{$spk2utt{$spk}})
  {
     $sid=1+int($i/$ups);
     if ($ups*$sid > $num)
     { 
       $sid-=1;
     }
     if ($sid < 10)
     {
       $sid="0$sid";
     }
    print fidout "$utt $spk-$sid\n";
    $i+=1;
  }
}
close(fidin);
close(fidout);
exit 0;