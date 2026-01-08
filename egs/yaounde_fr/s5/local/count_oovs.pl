#!/usr/bin/perl -W

# (c) 2014  Korbinian Riedhammer

# Count the number of OOV per turn (or speaker, if utt2spk is provided).  Use
# the --split-words option to split non-ascii words into characters (syllable
# based languages).


use strict;
use warnings;
use Getopt::Long;
use open qw(:std :utf8);


my $utt2spkf = "";
my $split_words = 0;

GetOptions(
	'utt2spk=s' => \$utt2spkf,
	'split-words' => \$split_words
);

if (scalar @ARGV lt 1) {
	print STDERR "usage:  $0 [--utt2spk=utt2spk] words.txt [input]\n";
	exit 1;
}

my $lexf = shift @ARGV;

my %lex = map { my ($a, $b) = split /\s+/; $a => $b; } `cat $lexf`;

my %utt2spk = ();
if (length $utt2spkf gt 0) {
	%utt2spk = map { my ($a, $b) = split /\s+/; $a => $b; } `cat $utt2spkf`; #read_file($utt2spkf, binmode => ':utf8');
}

my %num_words = ();
my %num_oovs = ();
my %oov_string = ();

while (<>) {
	my ($id, @trl) = split /\s+/;

	if (length $utt2spkf gt 0) {
		if (defined $utt2spk{$id}) {
			$id = $utt2spk{$id};
		} else {
			printf STDERR "Warning: $id not specified in $utt2spkf\n";
		}
	}

	$num_words{$id} = 0 unless defined $num_words{$id};
	$num_oovs{$id} = 0 unless defined $num_oovs{$id};
	$oov_string{$id} = ""  unless defined $oov_string{$id};


	if ($split_words) {
		for (my $i = 0; $i < scalar @trl; $i++) {
			my $w = $trl[$i];
			unless ($w =~ m/[a-zA-Z_\-]/) {
				my @sw = split //, $w;
				splice @trl, $i, 1, @sw;
				$i += (scalar @sw) - 1;
			}
		}
	}

	$num_words{$id} += scalar @trl;
	for my $w (@trl) {
		$num_oovs{$id} += 1 unless defined $lex{$w};
		$oov_string{$id} .= "$w " unless defined $lex{$w};
	}

}

for my $i (sort keys %num_words) {
	printf "%s %d %d %s\n", $i, $num_words{$i}, $num_oovs{$i}, 
		( defined $oov_string{$i} ? $oov_string{$i} : "");
}

