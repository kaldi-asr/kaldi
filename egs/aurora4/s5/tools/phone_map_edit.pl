#!/usr/bin/perl
#
open(IN,"$ARGV[0]") or die "input is phonemes seq,output is phoneme_id map\n";
open(OUT,">$ARGV[1]");

$num=1;
while(<IN>)
{
 chomp;
 s/\s+|\'|//g;
 if ($_ ne "")
 {
 print OUT $_." ".$num."\n";
 $num++;
 }

}
close IN;
close OUT;
