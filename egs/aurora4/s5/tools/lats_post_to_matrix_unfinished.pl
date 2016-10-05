#!/usr/bin/perl
#
#convert the *.lats.post file(vector for each utt) into a matrix for each utt(like the output of a nnet-forward output)
#
open(LAT,"$ARGV[0]") or die "input:lats.pdf.post, pdf_to_phone_map output:lats.phone.posts"; #lats.pdf.post: without utt id, starting with "[", and the posts all from lats.1.gz, not from best paths
open(MAP,"$ARGV[1]");# 42 phonemes(pdf-to-phone)
open(OUT,">$ARGV[2]");

$num_of_phonemes=42;

while(<MAP>)
{
 chomp;
 @arraymap=split/\s+/,$_;
 $hashmap{$arraymap[0]}=$arraymap[1]; #key:pdf-id value:phone-id 
}
close MAP;

$uttlat=0;
while(<LAT>)
{
 chomp;
 s/\]//g;
 @arraylat=split/\[/,$_; #[ is eliminated;
 #$uttid=shift @arraylat;# utt-id
 $num=@arraylat;
 @{$array_all_lat[$uttlat]}=@arraylat;
 ++$uttlat;
 print "post_on_lats contains $num +1 frames\n";
}
close LAT;

#@arrayphonepost=0*$num_of_phonemes;# 42 phones in total
$sum=0;

foreach $array_lat(@array_all_lat){
foreach $k(@$array_lat)
{
 $uttid=shift @array_lat;
 #if ($k ne "")
 #{
  @arraykey=split/\s+/,$k;
  @arraykey=&splice_array_empty(@arraykey);
  $numkkey=@arraykey;
  for (my $j=0;$j<$numkey;$j+=2)
  {
    $phone_id=$hashmap{$arraykey[$j]};
    $phone_post=$arraykey[$j+1];
    # if ($phone_post ne "1")
    #{
     
    #}
    print OUT 
  }
  #}
}
}


sub splice_array_empty()
{
 @array=@_;
 $num=@array;
 for ($i=0;$i<$num;++$i)
 {
  if ($array[$i] eq "")
  {
   splice @array,$i,1;
  }
 }
 return @array;
}
