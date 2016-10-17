#!/usr/bin/perl
use List::Util qw(min max);
use List::Util qw(sum);

#convert the *.lats.post file(vector for each utt) into a matrix for each utt(like the output of a nnet-forward output)
#
open(LAT,"$ARGV[0]") or die "input:lat.1.phone.post,map_root_phones_vs_dep_phones.int output:lats.phone.posts"; #lats.pdf.post: without utt id, starting with "[", and the posts all from lats.1.gz, not from best paths
open(MAP,"$ARGV[1]");# 42 phonemes(pdf-to-phone)
open(OUT,">$ARGV[2]");

$num_of_phonemes=42;

while(<MAP>)
{
 chomp;
 @arraymap=split/\s+/,$_;
 my $root=shift @arraymap;
 foreach my $dep_phone(@arraymap)
 {
  $hashmap{$dep_phone}=$root; #key:pdf-id value:phone-id 
 }
}
close MAP;

$uttlat=0;
while(<LAT>)
{
 chomp;
 s/\]//g;
 @arraylat=split/\[/,$_; #[ is eliminated;
 $uttid=shift @arraylat;# utt-id
 $num=@arraylat;
 @{$array_all_lat[$uttlat]}=@arraylat;
 ++$uttlat;
 print "post_on_lats contains $num  frames\n";
}
close LAT;
#print "utt num is $uttlat\n";
#@arrayphonepost=0*$num_of_phonemes;# 42 phones in total
$sum=0;

foreach $array_lat(@array_all_lat){ # per utterance
foreach $k(@$array_lat) # per frame
{
	#$uttid=shift @array_lat;
	#print "uttid is $uttid\n";
  @arraykey=split/\s+/,$k;
  @arraykey=&splice_array_empty(@arraykey);
  $numkey=@arraykey;
  # print "$numkey has posteriors not 0 \n";
  @array_phone_post=(0)x$num_of_phonemes;
  $phone_init_id=$hashmap{$arraykey[0]};
  $array_phone_post[$phone_init_id]=0;
  for (my $j=0;$j<$numkey;$j+=2)
  {
    $phone_id=$hashmap{$arraykey[$j]};
    $phone_post=$arraykey[$j+1];
    #print "dep phone is $arraykey[$j], phone is $phone_id\n";
    if ($phone_id eq $phone_init_id)
    {
      $array_phone_post[$phone_init_id-1]+=$phone_post; 
    }
    else
    {
      $phone_init_id=$phone_id;
      $array_phone_post[$phone_init_id-1]=$phone_post;
    } 
    #print "post id is $phone_id, post is $phone_post \n";
  }
  #@array_phone_post=&smooth_phone_post_array(@array_phone_post);
  $string_phone_post=join(' ', @array_phone_post);
  $string_phone_post=~ s/^\s+|\s+$//g;
  print OUT $string_phone_post."\n";
}
}
close OUT;
print "All done !\n";

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

sub smooth_phone_post_array() # write phone_post of each frame to a line in the whole matrix
{
  my @array=@_;
  my $max= max @array;

  my $epsilon=10 ** (-8);

  foreach my $key(@array)
  {
   if ($key eq 0)
   {
     $key=$epsilon;
   }
  }

  my $sum= sum @array;
  # print "sum is $sum\n";

  return @array;
}
