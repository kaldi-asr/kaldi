#!/usr/bin/perl

# Original Author: laurensw75
# Original source of this script: https://github.com/laurensw75/kaldi_egs_CGN.git

# takes a file list with lines containing only file names such as:
#   /Volumes/KALDI/CGN/wav/comp-f/nl/fn000042.wav
# it writes an utt2spk file, a spk2gender file, and also a txt file for transcriptions
#
# Its parameters are:
#  flist2scp <flist> <skpfile-directory> <speakerinfo> <utt2spk-file> <spk2gender-file> <txt-file> <segments-file>
#


use PerlIO::gzip;

#
# For each line in the flist, read the corresponding spk file.
#   - make segments file to indicate utterance borders
#   - make scp file with utterance-ids
#   - make utt2spk file to indicate speakerinformation
#   - make txt file with transcription of each utterance
# Then read the speakers.txt file for more info on speakers
#   - make spk2gender file with genderinformation
#

open(UTT, ">$ARGV[1].utt2spk");
open(TXT, ">$ARGV[1].txt");
open(SEG, ">$ARGV[1].segments");
open(SCP, ">$ARGV[1]_wav.scp");

open (IN, "$ARGV[1].flist");
while(<IN>){
	chop;
	m:^\S+/(.+)\.wav$: || next;
	$basefile=$1;
	$fullfile=$_;
	m:^\S+/(comp-.+)\.wav$:;
	$seafile=$1;

	# write SCP
	print SCP "$basefile sox -t wav $fullfile -b 16 -t wav - remix - |\n";

	# write SEA
	open(SEA, "$ARGV[0]/data/annot/corex/sea/$seafile.sea") || next;
	while (<SEA>) {
		chop;

		# Process start line
		if (m/^(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)/) {
			$uttid=$5;
			$speaker=$4;
			if ($speaker eq "COMMENT") {$speaker="BACKGROUND";}
			$end=$3/1000;
			$start=$2/1000;
			if($end<=$start) {$speaker="BACKGROUND";}
		}

		# Process orthographical transcription
		if ((m/^ORT\s(.+)(\?|\.)$/) && ($speaker ne "BACKGROUND")) {
			$line=$1;
			$line=~s/(\.*| *)$//g;
			$text=lc($line);			# use lowercase only
		}

		# Change non-words to <unk>, and write txt, seg, and utt files
		if ((m/^MAR\s(.+) _$/) && ($speaker ne "BACKGROUND")) {
			@words=split(" ", $text);
			@mars=split(" ", $1);

			if($#words==$#mars) {
				for ($t=0; $t<scalar(@words); $t++) {
					if ($mars[$t] ne "_") {
						$words[$t]="<unk>";
					}
				}
				$text=join(" ", @words);

				if (scalar(@words)>0) {
					# $text=~s/-/ /g;			# remove hyphens from words
         		$text=~s/  / /g;			# and remove double spaces
         		print TXT "$speaker-$uttid $text\n";
         		print SEG "$speaker-$uttid $basefile $start $end\n";
					$speakersfound{$speaker}=1;
					print UTT "$speaker-$uttid $speaker\n";
				}
			}
		}
	}
}

# Create spk2gender
# get gender for each speaker
open(IN, "$ARGV[0]/data/meta/text/speakers.txt") || die "speakers.txt not found";

# find indexes for ID and gender
$topline=<IN>;
@stuff=split(/\t/, $topline);
for($t=0; $t<$#stuff; $t++) {
	if ($stuff[$t] eq "ID") {
		$ididx=$t;
	} elsif ($stuff[$t] eq "sex") {
		$genderidx=$t;
	}
}

while(<IN>) {
	@stuff=split(/\t/);
	if ($stuff[$genderidx] eq "sex1") {
		$gender="m";
	} else {
		$gender="f";
	}
	$spk2gen{$stuff[$ididx]}=$gender;
}
$spk2gen{"UNKNOWN"}="m";		# unknowns are male

# and write speakers with gender to specified output
open(SPK, ">$ARGV[1].spk2gender");
foreach $spk (sort keys %speakersfound) {
    # some speakers are not in speakers.txt. We shall assume they are male for no particular reason whatsoever.
    if (!$spk2gen{$spk}) {
        $spk2gen{$spk}="m";
    }
    print SPK "$spk $spk2gen{$spk}\n";
}
