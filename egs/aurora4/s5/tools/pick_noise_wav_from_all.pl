#!/usr/bin/perl
#
#Available for wav&feats.scp;
#Need to compute_cmvn again

#open(ALL,"<$ARGV[0]"); 
open(CLEAN,"<$ARGV[1]");
open(NOISE,">$ARGV[2]");

while (<CLEAN>){
 chomp;
# print $_."\n";
 @tmp_clean=split/\s+/,$_;
 $key_clean=$tmp_clean[0];
 chop $key_clean;
# print "Utt is $key_clean \n";
 open(ALL,"<$ARGV[0]");
 while (<ALL>){
	 #print $_;
    chomp;
    $wav_all=$_;
	 if ( $wav_all=~ /$key_clean/ && $wav_all=~ /wv1/ && $wav_all=~ /street/ ){
		 print NOISE $wav_all."\n";
		 # print "Utt $key_clean is in $_\n";
	 }
	 else{
		 #print "Utt $key_clean is not in all list\n";
	 }
 }
 close ALL;
}
close CLEAN;
close NOISE;
#close ALL;
