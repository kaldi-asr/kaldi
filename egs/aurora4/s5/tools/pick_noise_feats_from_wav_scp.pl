#!/usr/bin/perl
#
#Available for wav&feats.scp;
#Need to compute_cmvn again

#open(ALL,"<$ARGV[0]"); 
open(WAV,"<$ARGV[1]");
open(PICKED,">$ARGV[2]");

while (<WAV>){
 chomp;
#print $_."\n";
 @tmp_wav=split/\s+/,$_;
 $key_wav=$tmp_wav[0];
 #chop $key_clean;
 #print "Utt is $key_wav \n";
 open(ALLFEAT,"<$ARGV[0]");
 while (<ALLFEAT>){
	 #print $_;
    chomp;
    @feat_all=split/\s+/,$_;
    $key_feat=$feat_all[0];
    #print $key_feat."\n";
	 if ( $key_feat eq $key_wav ){
		 print PICKED $_."\n";
		 # print "Utt $key_clean is in $_\n";
	 }
	 else{
		 # print "Utt $key_clean is not in all list\n";
	 }
 }
 close ALLFEAT;
}
close WAV;
close PICKED;
#close ALL;
