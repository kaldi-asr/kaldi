#!/usr/bin/perl
# Copyright     2020  Ivan Medennikov (STC-innovations Ltd)
# Apache 2.0.

# This script creates 3 negative utt2spk files with speakers from the same session.

($filein,$fileout,$fileout2,$fileout3)=@ARGV;

$Nspk=4;

%id2time2utt={};
%utt2spk={};
%utt2P={};
%sid2spk={};

open(fidin, "<$filein") or die "can't open $filein : $!";
while ($line=<fidin>)
{
  $line=~s/\s+$//;
  @items=split(/\s+/,$line);
  $utt=$items[0];
  $spk=$items[1];
  $utt2spk{$utt}=$spk;
  if ($utt=~/P(\d+)/)
  {
    $P=$1;
  }
  else
  {
    print "skipping utt $utt\n";
    next;
  }
  $utt2P{$utt}=$P;
  if ($utt=~/S(\d+)/)
  {
    $S=$1;
  }
  else
  {
    print "skipping utt $utt\n";
    next;
  }
  if ($utt=~/\D(\d{7})-\d{7}/)
  {
    $beg=$1;
  }
  else
  {
    print "skipping utt $utt\n";
    next;
  }
  $type=0;
  if ($utt=~/rev/)
  {
    $type=1;
  }
  elsif ($utt=~/\.L/)
  {
    $type=L;
  }
  elsif ($utt=~/\.R/)
  {
    $type=R;
  }
  elsif ($utt=~/(U\d+).+(CH\d+)/)
  {
    $type="$1_$2";
  }
  if ($utt=~/(sp0.9)/)
  {
    $type="$1_$type";
  }
  if ($utt=~/(sp1.1)/)
  {
    $type="$1_$type";
  }
  $id="S$S\_$type";
  if ( not exists $id2time2utt{$id} )
  {
    %{$id2time2utt{$id}}={};
  }
  push(@{$id2time2utt{$id}{$beg}},$utt);
  $sid="$P\_$type";
  push(@{$sid2spk{$sid}},$spk);
}
close(fidin);


open(fidout, ">$fileout") or die "can't open $fileout : $!";
open(fidout2, ">$fileout2") or die "can't open $fileout2 : $!";
open(fidout3, ">$fileout3") or die "can't open $fileout3 : $!";
foreach $id(sort keys %{id2time2utt})
{
  $type="";
  if ($id=~/S\d+\_(\S+)/)
  {
    $type=$1;
  }
  @utts=();
  %curspk={};
  foreach $time(sort keys %{$id2time2utt{$id}})
  {
    foreach $utt (@{$id2time2utt{$id}{$time}})
    {
    $P=$utt2P{$utt};
    if ($utt=~/^\s*$/) { next; }
    if (not exists $curspk{$P}) { $curspk{$P}=$utt2spk{$utt}; }
    push(@utts,$utt);
    }
  }
  foreach $utt (@utts)
  {
    $P=$utt2P{$utt};
    $curspk{$P}=$utt2spk{$utt};
    $Plast=int(($P-1)/$Nspk)*$Nspk+$Nspk;
    $P1=$P+1;
    if ($P1 > $Plast)
    {
      $P1=$P1-$Nspk;
    }
    if ($P1<10)
    {
      $P1="0$P1";
    }
    $P2=$P+2;
    if ($P2 > $Plast)
    {
      $P2=$P2-$Nspk;
    }
    if ($P2<10)
    {
      $P2="0$P2";
    }
    $P3=$P+3;
    if ($P3 > $Plast)
    {
      $P3=$P3-$Nspk;
    }
    if ($P3<10)
    {
      $P3="0$P3";
    }
    if ( not exists $curspk{$P1} ) { $sid=$P1."\_".$type; $cspk=$sid2spk{$sid}[rand @{$sid2spk{$sid}}]; print fidout "$utt $cspk\n";  }
    else { print fidout "$utt $curspk{$P1}\n"; }
    if ( not exists $curspk{$P2} ) { $sid=$P2."\_".$type; $cspk=$sid2spk{$sid}[rand @{$sid2spk{$sid}}]; print fidout2 "$utt $cspk\n"; }
    else { print fidout2 "$utt $curspk{$P2}\n"; }
    if ( not exists $curspk{$P3} ) { $sid=$P3."\_".$type; $cspk=$sid2spk{$sid}[rand @{$sid2spk{$sid}}]; print fidout3 "$utt $cspk\n"; }
    else { print fidout3 "$utt $curspk{$P3}\n"; }
  }
}

close(fidout);
close(fidout2);
close(fidout3);
exit 0;