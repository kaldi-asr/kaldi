#!/usr/bin/env perl
# Copyright 2014  Gaurav Kumar.   Apache 2.0

use File::Basename;
($tmpdir)=@ARGV;
#$tmpdir='../data/local/tmp';
$trans="$tmpdir/train_transcripts.flist";
$reco="$tmpdir/reco2file_and_channel";
open(T, "<", "$trans") || die "Can't open transcripts file";
open(R, "|sort >$reco") || die "Can't open reco2file_and_channel file $!";
open(O, ">$tmpdir/text.1") || die "Can't open text file for writing";
open(G, ">$tmpdir/spk2gendertmp") || die "Can't open the speaker to gender map file";
while (<T>) {
	$file = $_;
	m:([^/]+)\.tdf: || die "Bad filename $_";
	$call_id = $1;
	print R "$call_id-A $call_id A\n";
	print R "$call_id-B $call_id B\n";
	open(I, "<$file") || die "Opening file $_";
	# Get rid of header sections first
	foreach ( 0..2 ) {
		$tmpLine = <I>;
	}	
	#Now read each line and extract information
	while (<I>) {
		#20051017_215732_274_fsp.sph     1       0.0     0.909856781803  Audrey  female  native   <foreign lang="English"> Audrey </foreign>     0       0       -1	
		chomp;
		my @stringComponents = split(/\t/);
		
		#Check number of components in this array
		if ((scalar @stringComponents) >= 11) {
			$start = sprintf("%06d", $stringComponents[2] * 100);
			$end = sprintf("%06d", $stringComponents[3] * 100);
			length($end) > 6 && die "Time too long $end in $file";
			$side = $stringComponents[1] ? "B" : "A";
			$words = $stringComponents[7];
			$utt_id = "${call_id}-$side-$start-$end";
			$speaker_id = "${call_id}-$side";
			$gender = "m";
			if  ($stringComponents[5] == "female") {
				$gender = "f";
			}
			print G "$speaker_id $gender\n" || die "Error writing to speaker2gender file";
			$words =~ s:</:lendarrow:g;
			$words =~ s/</larrow/g;
			$words =~ s/>/rarrow/g;
			$words =~ s/[[:punct:]]//g;
			$words =~ s/larrow/</g;
			$words =~ s/rarrow/>/g;
			$words =~ s:lendarrow:</:g;
			$words =~ s/Á/á/g;
			$words =~ s/Í/í/g;
			$words =~ s/Ó/ó/g;
			$words =~ s/Ú/ú/g;
#			$words =~ s/ì/í/g;
#			$words =~ s/è/é/g;
#			$words =~ s/¡/i/g;
#			$words =~ s/J/J/g;
#			$words =~ s/S/S/g;
#			$words =~ s/à/á/g;
			$words =~ s/¨//g;
			$words =~ s/·//g;
			$words =~ s/´//g;
            $words =~ s/N/n/g;
#			$words =~ s/2//g;
			$words = lc($words);
			$words =~ s:ü([eiéí]):w\1:g;
			$words =~ s:ü:u:g;
			$words =~ s:ñ:N:g;
			print O "$utt_id $words\n" || die "Error writing to text file";
		}
	}
}
close(T);
close(R);
close(O);
close(G);
