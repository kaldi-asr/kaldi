#!/usr/bin/perl -w
#usma_nonnative_copy_wav_files.pl - copy and convert   wav files to 16000HZ
# writes files under data/local/tmp/usma/nonnative/wavs

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

my $tmpdir = "data/local/tmp/usma/nonnative";

# input wav file list
my $w = "$tmpdir/wav_list.txt";

# output temporary wav.scp files
my $o = "$tmpdir/wav.scp";

# output temporary utt2spk files
my $u = "$tmpdir/utt2spk";

# output temporary text files
my $t = "$tmpdir/text";

# initialize hash for prompts
my %p = ();

open my $P, '<', $p or croak "problem with $p $!";

# store prompts in hash
LINEA: while ( my $line = <$P> ) {
    chomp $line;
    my ($num,$sent) = split /\t/, $line, 2;
    # remove the leading s
    $num =~ s/s(.+)$/$1/;

    # format the utterance number to 3 digits
    if ( $num < 10 ) {
	$num = 0 . 0 . $num;
    } elsif ( $num < 100 ) {
	$num = 0 . $num;
    }

    $p{$num} = $sent;
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

	next LINE unless ( $line =~ /nonnative/ );

	my ($volume,$directories,$file) = File::Spec->splitpath( $line );

	my @dirs = split /\//, $directories;

	my $r = basename $line, ".wav";

	next LINE unless ( $r =~ /^s/ );

	my $s = $dirs[-1];
	my ($nativeness,$gender,$country,$weight,$age,$height,$dlpt,$idx) = split /\-/, $s, 9;

	my $rid = "";

	# make the file basename have 3 digit places
	$r =~ s/^s(.+)/$1/;
	if ( $r < 10 ) {
	    $r = 0 . 0 . $r;
	} elsif ( $r < 100 ) {
	    $r = 0 . $r;
	}

	$s = $nativeness . '_' . $gender . '_' . $country . '_' . $weight . '_' . $age . '_' . $height . '_' . $dlpt . '_' . $idx;
	$rid = $s . '_' . $r;

	if ( ! -d "$tmpdir/wavs/audio/$s" ) {
	    system "mkdir -p $tmpdir/wavs/audio/$s";
	}

	if ( ! -d "$tmpdir/wavs/transcription/$s" ) {
	    system "mkdir -p $tmpdir/wavs/transcription/$s";
	}

	if ( exists $p{$r} ) {
	    print $T "$rid\t$p{$r}\n";
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
	print $X "$p{$r}\n";
	close $X;
    }
    close $T;
    close $O;
    close $U;
    close $W;
    return 1;
}
