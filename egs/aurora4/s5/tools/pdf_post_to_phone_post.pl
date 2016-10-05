#!/usr/bin/perl
#
#This script now can only read one-line ali.pdf.file
#
$pdf_ali=$ARGV[0]; # pdf id sequences; getting from ali-to-pdf; ali.pdf file
$pdf_phone_tab=$ARGV[1];
$phone_id_to_phone_tab=$ARGV[2];
$output=$ARGV[3];

if (@ARGV !=4)
{
  print "Usage: input is <pdf.ali> <pdf-to-phone-map> <phone-id-to-sym-map>; output is phone-id-seq\n";
}
open(PDFTAB,"<$pdf_phone_tab");#or die "pdf_ali pdf_to_phone_tab phoneid2phone_tab output\n";
open(ALI,"<$pdf_ali"); # starting with uttid 
open(PHONE,"<$phone_id_to_phone_tab"); # pseduo_phone
open(OUT,">$output");# output is phoneme sequences 

%hashpdf2phone=();

while (<PDFTAB>)
{
 chomp;
 @array=split/\s+/,$_;
 $hashpdf2phone{$array[0]}=$array[1];
 @array=();
}
close PDFTAB;

@newarray=();

$arraydim=0;
while (<ALI>)
{
 chomp;
 @array2=split/\s+/,$_;
 $uttid=shift @array2;
 $num=@array2;
 push @{ $newarray[$arraydim] },$uttid;
 for ($i=0;$i<$num;++$i)
 {
	 # if (($array2[$i] ne "\[") && ($array2[$i] ne "\]"))
	 #{
		 push @{ $newarray[$arraydim]},$hashpdf2phone{$array2[$i]};
		 #}
  #print $newarray[$i]."\n";
 }
 #push @{ $newarray[$arraydim] }," ]";
# push @{$newwarray[$arraydim]}," ]";
$arraydim++;
}

close ALI;

$size=@newarray;
print "number of utts is $size\n";
%hashphone=();
while (<PHONE>)
{
 chomp;
 @arrayphone=split/\s+/,$_;
 #$_=$arrayphone[0];
 if ($arrayphone[0]=~ /\_/)
{
 @array3=split/\_/,$arrayphone[0];
 $arrayphone[0]=$array3[0];
}
$hashphone{$arrayphone[1]}=$arrayphone[0];
}
close PHONE;

#print OUT "$uttid ";
foreach my $dim_1(@newarray)
{
	
	$string=join " ",@{$dim_1};
#	foreach my $dim_2( @{$dim_1} )
#	{
#		print OUT $dim_2." ";
#	}
	print OUT $string."\n";
}
close OUT;
#
#for ($j=0;$j<$num;++$j)
#{
# 
#	print OUT $newarray[$j]." ";
#	#print OUT $hashphone{$newarray[$j]}." ";
#}
##print OUT @newarray;
##close OUT;
