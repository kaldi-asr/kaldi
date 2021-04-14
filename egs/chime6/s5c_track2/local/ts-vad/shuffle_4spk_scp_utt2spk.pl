#!/usr/bin/perl
# Copyright     2020  Ivan Medennikov (STC-innovations Ltd)
# Apache 2.0.

# This script takes 4 scp files $filein{1,2,3,4} with the same utterance-ids, 
# and produces 4 shuffled versions of them.
# Moreover, the same shuffling is performed with 4 utt2spk files $utt2spk{1,2,3,4}.

use List::Util qw(shuffle);

($filein1,$filein2,$filein3,$filein4,$fileout1,$fileout2,$fileout3,$fileout4,$utt2spk1,$utt2spk2,$utt2spk3,$utt2spk4,$out1,$out2,$out3,$out4)=@ARGV;

%utt2arks={};
%utt2spk1={};
open(fidin, "<$utt2spk1") or die "can't open $utt2spk1 : $!";
while ($line=<fidin>)
{
  $line=~s/\s+$//;
  @items=split(/\s+/,$line);
  $utt=$items[0];
  $spk=$items[1];
  $utt2spk1{$utt}=$spk;
}
close(fidin);

%utt2spk2={};
open(fidin, "<$utt2spk2") or die "can't open $utt2spk2 : $!";
while ($line=<fidin>)
{
  $line=~s/\s+$//;
  @items=split(/\s+/,$line);
  $utt=$items[0];
  $spk=$items[1];
  $utt2spk2{$utt}=$spk;
}
close(fidin);

%utt2spk3={};
open(fidin, "<$utt2spk3") or die "can't open $utt2spk3 : $!";
while ($line=<fidin>)
{
  $line=~s/\s+$//;
  @items=split(/\s+/,$line);
  $utt=$items[0];
  $spk=$items[1];
  $utt2spk3{$utt}=$spk;
}
close(fidin);

%utt2spk4={};
open(fidin, "<$utt2spk4") or die "can't open $utt2spk4 : $!";
while ($line=<fidin>)
{
  $line=~s/\s+$//;
  @items=split(/\s+/,$line);
  $utt=$items[0];
  $spk=$items[1];
  $utt2spk4{$utt}=$spk;
}
close(fidin);


open(fidin, "<$filein1") or die "can't open $filein1 : $!";
while ($line=<fidin>)
{
  $line=~s/\s+$//;
  @items=split(/\s+/,$line, 2);
  $utt=$items[0];
  $ark=$items[1];
  push(@{$utt2arks{$utt}},"$utt2spk1{$utt} $ark");
}
close(fidin);

open(fidin, "<$filein2") or die "can't open $filein2 : $!";
while ($line=<fidin>)
{
  $line=~s/\s+$//;
  @items=split(/\s+/,$line, 2);
  $utt=$items[0];
  $ark=$items[1];
  push(@{$utt2arks{$utt}},"$utt2spk2{$utt} $ark");
}
close(fidin);

open(fidin, "<$filein3") or die "can't open $filein3 : $!";
while ($line=<fidin>)
{
  $line=~s/\s+$//;
  @items=split(/\s+/,$line, 2);
  $utt=$items[0];
  $ark=$items[1];
  push(@{$utt2arks{$utt}},"$utt2spk3{$utt} $ark");
}
close(fidin);

open(fidin, "<$filein4") or die "can't open $filein4 : $!";
while ($line=<fidin>)
{
  $line=~s/\s+$//;
  @items=split(/\s+/,$line, 2);
  $utt=$items[0];
  $ark=$items[1];
  push(@{$utt2arks{$utt}},"$utt2spk4{$utt} $ark");
}
close(fidin);

open(fidout1, ">$fileout1") or die "can't open $fileout1 : $!";
open(fidout2, ">$fileout2") or die "can't open $fileout2 : $!";
open(fidout3, ">$fileout3") or die "can't open $fileout3 : $!";
open(fidout4, ">$fileout4") or die "can't open $fileout4 : $!";

open(out1, ">$out1") or die "can't open $out1 : $!";
open(out2, ">$out2") or die "can't open $out2 : $!";
open(out3, ">$out3") or die "can't open $out3 : $!";
open(out4, ">$out4") or die "can't open $out4 : $!";

foreach $utt(sort(keys %utt2arks))
{
  if (scalar(@{$utt2arks{$utt}}) < 4)
  {
    next;
  }
  @shf = shuffle(@{$utt2arks{$utt}});
  @u1 = split(/\s+/, $shf[0], 2);
  @u2 = split(/\s+/, $shf[1], 2);
  @u3 = split(/\s+/, $shf[2], 2);
  @u4 = split(/\s+/, $shf[3], 2);

  print fidout1 "$utt $u1[1]\n";
  print fidout2 "$utt $u2[1]\n"; 
  print fidout3 "$utt $u3[1]\n";
  print fidout4 "$utt $u4[1]\n";

  print out1 "$utt $u1[0]\n";
  print out2 "$utt $u2[0]\n";
  print out3 "$utt $u3[0]\n";
  print out4 "$utt $u4[0]\n";

}
close(fidout1);
close(fidout2);
close(fidout3);
close(fidout4);

close(out1);
close(out2);
close(out3);
close(out4);

exit 0;

