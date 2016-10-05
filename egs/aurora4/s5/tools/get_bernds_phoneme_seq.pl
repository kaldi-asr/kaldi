#!/usr/bin/perl
#
#convert nbest.lats.phonemes.n(with phonemes) to ~ with phonemes-id seqs according to Bernd's matlab phone-id map
#

open(MAP,$ARGV[0]) or die "input is Bernd's Matlab phone-id map and nbest.lats.phonemes.n; output is same format with phone-id seq\n";

open(IN,$ARGV[1]);

open(OUT,">$ARGV[2]");

while(<MAP>)
{
 chomp;
 @array=split/\s+/,$_;
 $hashmap{$array[0]}=$array[1];
 #print $array[1]."\n";
}
close MAP;

while(<IN>)
{
 chomp;
 @array2=split/\s+/,$_;
}
close IN;

foreach $k(@array2)
{
	print "$k\n";
 print OUT $hashmap{$k}." ";
}
close OUT;

