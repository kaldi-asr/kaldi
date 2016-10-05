#!/usr/bin/perl
#
#This script convert nbest.pdf.post.* file to phone post matrix(num_phonemes * frames)
#Difference from lats_post_to_matrix is this one take in a vector with only pdf-posts( no pdf-ids), and it's a best path through the lats, so only one post per frame.
#
#As for KL Divergence, get average post for left phonemes (1-max)/num_of_left_phonemes
#
#print @ARGV."\n";
 if ( @ARGV != 4)
 {
	print "Usage: intput is <nbest.phoneme.id>, <nbest.pdf.post>,<output-dir>, <num-of-phonemes>,output is nbest.phone.post.matrix\n ";  
  exit ;
 }

open(PHONEID,"<$ARGV[0]");# or die a seq of phone-ids; totally 42 phones
open(PDFPOST,"<$ARGV[1]"); # a seq of pdf-posts with uttid and [ ]
$outdir=$ARGV[2];
$total_phonemes=$ARGV[3];
 

$utt_post=0;
$utt_phones=0;
while(<PHONEID>)
{
 chomp;
 @arrayphone=split/\s+/,$_;
 @{$array_sum_phone[$utt_phones]}=@arrayphone;
 ++$utt_phones;
}
close PHONEID;
#print "size of phone-id contains @array_sum_phone utts\n";

@arraymatrix=(0)x$total_phonemes;

#print @arraymatrix."\n";
$dim=0;
while(<PDFPOST>)
{
 chomp;
 @arraypdf=split/\s+/,$_;
 $uttid=shift @arraypdf;#uttid
 shift @arraypdf;#[
 pop @arraypdf;#]
 unshift @arraypdf,$uttid;
 @{$array_sum_post[$utt_post]}=@arraypdf;
 ++$utt_post;
}
close PDFPOST;

$avg=0;
$index=0;

$utt_post=@array_sum_post;
$utt_phones=@array_sum_phone;
print "utt in post:$utt_post\n";
print "utt in phones:$utt_phones\n";

if ($utt_post ne $utt_phones)
{
	print "Utts num in pdf.post vs. phone.id mismatch!\n";
	exit ;
}

#for ($i=0;$i<82;++$i)
#{
#	@dim_1_post=@{$array_sum_post[$i]};
#	print $i.": ".$dim_1_phone[0]."\n";
#         print $array_sum_post[$i]->[0]."\n"; 
#}

for ($utt=0;$utt<$utt_post;++$utt)
{
	$num_post=@{$array_sum_post[$utt]};
	$utt_id_post=$array_sum_post[$utt][0];
#	print "frames for $utt_id_post : $num_post\n";
	$num_phones=@{$array_sum_phone[$utt]};
	$utt_id_phones=$array_sum_phone[$utt][0];
	# print "$utt_id_post vs. $utt_id_phones.\n";
	if ($utt_id_post ne $utt_id_phones)
	{
		       print "$utt_id in post vs. phone-id mismatch\n";
			exit ;
	}
	elsif ( $num_post ne $num_phones) 
		{
                	print "frames in $utt_id_post(post) vs. $utt_phones(phones) mismatch\n";
			exit ;
		}
		else{
		       $outfile=$outdir."\/".$utt_id_post.".post.matrix";
		print $outfile."\n";	
		       open(OUT,">$outfile");
			for ($frame=1;$frame < $num_post;++$frame)
		 {
			 #	 print "frame $frame in $utt_post\n";
			$pdf=$array_sum_post[$utt][$frame];
			if ($pdf ne "1"){
				$avg=(1-$pdf)/($total_phonemes-1);
				#	@arraymatrix=($avg)x$total_phonemes;
				#$offset=$array_sum_phone[$utt][$frame];
				#$arraymatrix[$offset]=$pdf;
				#print OUT $arraymatrix[$offset]."\n";
			}
			else{ 
			       	$pdf=0.9999999;
				$avg=(1-$pdf)/($total_phonemes-1);
			}
			        @arraymatrix=($avg)x$total_phonemes;
				$offset=$array_sum_phone[$utt][$frame];
				$arraymatrix[$offset]=$pdf;
				#print OUT $arraymatrix[$offset]."\n";
			#unshift @arraymatrix," ";
		       	$matrixline=join " ",@arraymatrix;
			$matrixline=~ s/^\s+ | \s+$//g;
		       	print OUT $matrixline."\n";
		}
		print "utt ($utt+1) is done\n";
		close OUT;
	}
	#close OUT;
}
