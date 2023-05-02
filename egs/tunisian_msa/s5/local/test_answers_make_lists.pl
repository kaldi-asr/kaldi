#!/usr/bin/env perl

# Copyright 2018 John Zac76
# Apache 2.0.

# test_answers_make_lists.pl - make acoustic model training lists

use strict;
use warnings;
use Carp;

use File::Spec;
use File::Copy;
use File::Basename;

BEGIN {
    @ARGV == 3 or croak "USAGE $0 <TRANSCRIPT_FILENAME> <SPEAKER_NAME> <COUNTRY>
example:
$0 /home/zak76/Desktop/Kaldi/kaldi-master/tunisian_msa-master/Libyan_collected_test/TEST/Libyan_MSA/adel/data/transcripts/answers/adel_answers.tsv adel libyan
";
}

my ($tr,$spk,$l) = @ARGV;

open my $I, '<', $tr or croak "problems with $tr";

my $tmp_dir = "data/local/tmp/$l/$spk";

system "mkdir -p $tmp_dir/answers";

# input wav file list
my $w = "$tmp_dir/answers_wav.txt";

# output temporary wav.scp files
my $o = "$tmp_dir/answers/wav.scp";

# output temporary utt2spk files
my $u = "$tmp_dir/answers/utt2spk";

# output temporary text files
my $t = "$tmp_dir/answers/text";

# initialize hash for prompts
my %p = ();

# store prompts in hash
LINEA: while ( my $line = <$I> ) {
    chomp $line;
    my ($s,$sent) = split /\t/, $line, 2;
    $p{$s} = $sent;
}

open my $W, '<', $w or croak "problem with $w $!";
open my $O, '+>', $o or croak "problem with $o $!";
open my $U, '+>', $u or croak "problem with $u $!";
open my $T, '+>', $t or croak "problem with $t $!";

 LINE: while ( my $line = <$W> ) {
     chomp $line;
     next LINE if ($line =~ /recordings/ );
     next LINE unless ( $line =~ /answers/ );
     my ($volume,$directories,$file) = File::Spec->splitpath( $line );
     my @dirs = split /\//, $directories;
     my $b = basename $line, ".wav";
     my ($sk,$r) = split /\_/, $b, 2;
     my $s = $dirs[-1];
     my $rid = $sk . '_' . $r;
     if ( exists $p{$b} ) {
	 print $T "$rid\t$p{$b}\n";
     } elsif ( defined $rid ) {
	 warn  "problem\t$rid";
	 next LINE;
     } else {
	 croak "$line";
     }

     print $O "$rid sox $line -t wav - |\n";
	print $U "$rid\t${sk}_a\n";
}
close $T;
close $O;
close $U;
close $W;
