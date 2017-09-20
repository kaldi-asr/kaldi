#!/usr/bin/perl -w
# heroico_answers_copy_wav_files.pl - copy and convert   wav files to 16000hz 
# writes files under data/local/tmp/heroico/wavs

use strict;
use warnings;
use Carp;

use File::Spec;
use File::Copy;
use File::Basename;

BEGIN {
    @ARGV == 1 or croak "USAGE: $0 <PROMPTSLIST>";
}

my ($p) = @ARGV;

my $tmpdir = "data/local/tmp/heroico";

system "mkdir -p $tmpdir/answers";

# input wav file list
my $w = "$tmpdir/wav_list.txt";

# output temporary wav.scp files
my $o = "$tmpdir/answers/wav.scp";

# output temporary utt2spk files
my $u = "$tmpdir/answers/utt2spk";

# output temporary text files
my $t = "$tmpdir/answers/text";

# initialize hash for prompts
my %p = ();

open my $P, '<', $p or croak "problem with $p $!";

# store prompts in hash
LINEA: while ( my $line = <$P> ) {
    chomp $line;
    my ($num,$sent) = split /\t/, $line, 2;
    my ($volume,$directories,$file) = File::Spec->splitpath( $num );
    my @dirs = split /\//, $directories;
    # get the speaker number
    my $s = $dirs[-1];

    my $i = "";

    # format the utterance number to 3 digits
    if ( $file < 10 ) {
	$file = 0 . 0 . $file;
    } elsif ( $file < 100 ) {
	$file = 0 . $file;
    }

    # format the speaker number to 3 digits
    if ( $s< 10) {
	$s = 0 . 0 . $s;
    } elsif ( $s < 100 ) {
	$s = 0 . $s;
    }

    # the utterance name
    $i = $s . '_' . $file . '_a';

    $p{$i} = $sent;
}
close $P;

prep($w,$o,$u,$t);

sub prep {
    my ($w,$o,$u,$t) = @_;

    my $md = "";
    my $language = "";

    open my $W, '<', $w or croak "problem with $w $!";
    open my $O, '+>', $o or croak "problem with $o $!";
    open my $U, '+>', $u or croak "problem with $u $!";
    open my $T, '+>', $t or croak "problem with $t $!";

    LINE: while ( my $line = <$W> ) {
	chomp $line;
	next LINE unless ( $line =~ /Answers/ );
	next LINE if ( $line =~ /Recordings/ );

	my ($volume,$directories,$file) = File::Spec->splitpath( $line );
	my @dirs = split /\//, $directories;
	my $r = basename $line, ".wav";

	my $s = $dirs[-1];
	my $rid = "";

	# make the speaker id have 3 digit places
	if ( $s < 10 ) {
	    $s = 0 . 0 . $s;
	} elsif ( $s < 100 ) {
	    $s = 0 . $s;
	} elsif ( $s >= 100 ) {
	    # we have less than 1000 speakers
	    $s = $s;
	}

	    # make the file basename have 3 digit places
	if ( $r < 10 ) {
	    $r = 0 . 0 . $r;
	} elsif ( $r < 100 ) {
	    $r = 0 . $r;
	}

	$rid = $s . '_' . $r . '_a';

	if ( ! -d "$tmpdir/wavs/audio/$s" ) {
	    system "mkdir -p $tmpdir/wavs/audio/$s";
	}

	if ( ! -d "$tmpdir/wavs/transcription/$s" ) {
	    system "mkdir -p $tmpdir/wavs/transcription/$s";
	}

	if ( exists $p{$rid} ) {
	    print $T "$rid\t$p{$rid}\n";
	} elsif ( defined $rid ) {
	    warn  "problem\t$rid";
	    next LINE;
	} else {
	    croak "$line";
	}

	print $O "$rid\t$tmpdir/wavs/audio/$s/$rid.wav\n";
	print $U "$rid\t$s\n";
	system "sox \\
-r 22050 \\
-e signed \\
-b 16 \\
$line \\
-r 16000 \\
$tmpdir/wavs/audio/$s/$rid.wav";
	open my $X, '+>', "$tmpdir/wavs/transcription/$s/$rid.txt" or croak "$!";
	print $X "$p{$rid}\n";
	close $X;
    }
    close $T;
    close $O;
    close $U;
    close $W;
    return 1;
}
