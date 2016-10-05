#!/usr/bin/perl
#use List::Util qw/max min/;
#use List::MoreUtils qw(first_index);
#
$post=$ARGV[0];
$phone_tab=$ARGV[1];#a table with phoneme and phoneme-ids
$output1=$ARGV[2];# phone seq;
$output2=$ARGV[3];#post seq;

open(POST,"<$post") or die "phone_post_from_nnet2 phone_id_map output_phone_seq output_post_seq\n";
open(TAB,"<$phone_tab");
open(PHONE,">$output1");
open(OUT,">$output2");
open(TMP,">phone.tab");

while (<TAB>)
{
 chomp;
 @tab=split/\s+/,$_;
 if ($tab[0]=~ /\_/)
 {
  @array1=split/\_/,$tab[0];
  $tab[0]=$array1[0];
 }
 $hashtab{$tab[1]}=$tab[0];
# print $hashtab{$tab[1]}." ".$tab[1]."\n";
}
close TAB;

while($line=<POST>)
{
 chomp $line;
 if ($line=~ /\[/)
 {
	 # print $line." \n";
  print OUT $line." ";
  next;
 }
 @array=split/\s+/,$line;
 my $num=@array;
 #print "dim is $num\n";
 if ($array[0] eq "" or $array[0] eq " "){
 shift @array;# first element is empty;
 }
 #print "array contains $num \n";
 #my $index=first_index {$_ eq '$max'} @array;
 $max_index=&arraymax(@array); 
 print "max index is ".$max_index."\n";
#my $num=$#array;
 #my $max=$i;
 #$max=$array[$i]> $array[$max] ? $i:$max while $i--;
 print PHONE $hashtab{$max_index}." ";
 print "max prob phone is $hashtab{$max_index}\n";
 print OUT $array[$max_index]." ";
# print "id is $max_index, phone is "
}
#print "dim is $num\n";
close POST;
close PHONE;
close OUT;

sub arraymax
{
 my @array=@_;
 my $max=$array[0];
 my $index=0;
 my $num=-1;
 foreach my $j (@array){
	 $num++;
	 if ($j > $max )
	 {
		 $max=$j;
		 $index=$num;
	 }
 }
 return $index;
}
