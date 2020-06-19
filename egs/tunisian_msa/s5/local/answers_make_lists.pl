#!/usr/bin/env perl

# Copyright 2018 John Morgan
# Apache 2.0.

# answers_make_lists.pl - make acoustic model training lists

use strict;
use warnings;
use Carp;

use File::Spec;
use File::Copy;
use File::Basename;

my $tmpdir = 'data/local/tmp/tunis';

system "mkdir -p $tmpdir/answers";

# input wav file list
my $wav_list = "$tmpdir/answers_wav.txt";

# output temporary wav.scp files
my $wav_scp = "$tmpdir/answers/wav.scp";

# output temporary utt2spk files
my $u = "$tmpdir/answers/utt2spk";

# output temporary text files
my $t = "$tmpdir/answers/text";

# initialize hash for prompts
my %prompt = ();

# store prompts in hash
LINEA: while ( my $line = <> ) {
    chomp $line;
    my ($num,$sent) = split /\t/sxm, $line, 2;

    my ($machine,$s,$mode,$language,$i) = split /\_/sxm, $num;
    # the utterance name
    my $utt = $machine . '_' . $s . '_' . 'a' . '_' . $i;
    $prompt{$utt} = $sent;
}

# Write wav.scp, utt2spk and text files.
open my $W, '<', $wav_list or croak "problem with $wav_list $!";
open my $O, '+>', $wav_scp or croak "problem with $wav_scp $!";
open my $U, '+>', $u or croak "problem with $u";
open my $T, '+>', $t or croak "problem with $t";

 LINE: while ( my $line = <$W> ) {
     chomp $line;
     next LINE if ( $line !~ /Answers/sxm );
     next LINE if ( $line =~ /Recordings/sxm );
     my ($volume,$directories,$file) = File::Spec->splitpath( $line );
     my @dirs = split /\//sxm, $directories;
     my $r = basename $line, '.wav';
     my $machine = $dirs[-3];
     my $s = $dirs[-1];
     my $rid = $machine . '_' . $s . '_' . 'a' . '_' . $r;
     if ( exists $prompt{$rid} ) {
	 print ${T} "$rid\t$prompt{$rid}\n" or croak;
     } elsif ( defined $rid ) {
	 print STDERR "problem\t$rid" or croak;
	 next LINE;
     } else {
	 croak "$line";
     }

	print ${O} "$rid sox $line -t wav - |\n" or croak;
     print ${U} "$rid ${machine}_${s}_a\n" or croak;
}
close $U or croak;
close $T or croak;
close $W or croak;
close $O or croak;
