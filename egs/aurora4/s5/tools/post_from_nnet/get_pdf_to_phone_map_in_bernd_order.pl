#!/usr/bin/perl
#
open(PDFMAP,"<$ARGV[0]") or die "input is pdf-phone.map in kaldi order, and phone_in_kaldi_vs_phone_bernd.map, output is pdf-phome.map in Bernd order\n";

open(PHONEMAP,"<$ARGV[1]");#1st col:bernd-id; #2nd col:kaldi-order
open(OUTPUT,">$ARGV[2]");

while(<PHONEMAP>)
{
 chomp;
 @array=split/\s+/,$_;
 $hashphone{$array[1]}=$array[0]; # key is kaldi, value is bernd
}
close PHONEMAP;

while(<PDFMAP>)
{
 chomp;
 @arraypdf=split/\s+/,$_;
 print OUTPUT $arraypdf[0]." ".$hashphone{$arraypdf[1]}."\n";
}
close PDFMAP;
close OUTPUT;

