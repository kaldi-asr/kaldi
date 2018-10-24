#!/usr/bin/env perl

# Copyright 2018 John Morgan
# Apache 2.0.

# devtest_recordings_make_lists.pl - make acoustic model training lists

use strict;
use warnings;
use Carp;

use File::Spec;
use File::Copy;
use File::Basename;

BEGIN {
    @ARGV == 3 or croak "USAGE $0 <TRANSCRIPT_FILENAME> <SPEAKER_NAME> <COUNTRY>
example:
$0 Tunisian_MSA/data/transcripts/devtest/recordings.tsv 6 tunisia
";
}

my ($tr,$spk,$l) = @ARGV;

open my $I, '<', $tr or croak "problems with $tr";

my $tmp_dir = "data/local/tmp/$l/$spk";

# input wav file list
my $wav_list = "$tmp_dir/wav.txt";
croak "$!" unless ( -f $wav_list );
# output temporary wav.scp files
my $wav_scp = "$tmp_dir/wav.scp";

# output temporary utt2spk files
my $u = "$tmp_dir/utt2spk";

# output temporary text files
my $t = "$tmp_dir/text";

# initialize hash for prompts
my %p = ();

# store prompts in hash
LINEA: while ( my $line = <$I> ) {
    chomp $line;
    my ($s,$sent) = split /\t/, $line, 2;
    $p{$s} = $sent;
}

open my $W, '<', $wav_list or croak "problem with $wav_list $!";
open my $O, '+>', $wav_scp or croak "problem with $wav_scp $!";
open my $U, '+>', $u or croak "problem with $u $!";
open my $T, '+>', $t or croak "problem with $t $!";

 LINE: while ( my $line = <$W> ) {
     chomp $line;
     next LINE if ($line =~ /answers/ );
     next LINE unless ( $line =~ /Recordings/ );
     my ($volume,$directories,$file) = File::Spec->splitpath( $line );
     my @dirs = split /\//, $directories;
     my $b = basename $line, ".wav";
     my $s = $dirs[-1];
     my $rid = $s . '_' . 'recording' . '_' . $b;
     my $uid = $s . '_' . 'recording';
     if ( exists $p{$b} ) {
	 print $T "$rid\t$p{$b}\n";
     } elsif ( defined $s ) {
	 warn  "problem\t$s";
	 next LINE;
     } else {
	 croak "$line";
     }

     print $O "$rid sox $line -t wav - |\n";
	print $U "$rid\t$uid\n";
}
close $T;
close $O;
close $U;
close $W;
