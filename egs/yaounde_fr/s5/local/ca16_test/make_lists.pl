#!/usr/bin/env perl

# Copyright 2018 John Morgan
# Apache 2.0.

# make_lists.pl - write lists for acoustic model training
# writes files under data/local/tmp/ca16_test/lists

use strict;
use warnings;
use Carp;

BEGIN {
    @ARGV == 1 or croak "USAGE: $0 <DATA_SRC_DIR>
Example:
$0 African_Accented_French
";
}

use File::Spec;
use File::Copy;
use File::Basename;

my ($d) = @ARGV;

# Initialize variables
my $tmpdir = "data/local/tmp/ca16_test";
my $p = "$d/transcripts/test/ca16/prompts.txt";
# input wav file list
my $w = "$tmpdir/wav_list.txt";
# output temporary wav.scp files
my $o = "$tmpdir/lists/wav.scp";
# output temporary utt2spk files
my $u = "$tmpdir/lists/utt2spk";
# output temporary text files
my $t = "$tmpdir/lists/text";
# initialize hash for prompts
my %p = ();
# done setting variables

system "mkdir -p $tmpdir/lists";

open my $P, '<', $p or croak "problem with $p $!";

# store prompts in hash
LINEA: while ( my $line = <$P> ) {
  chomp $line;
  my ($j,$sent) = split /\s/, $line, 2;
  # down case
  $sent = lc $sent;
  # no punctuation
  $sent =~ s/[\.,;:?]//g;
  # tokenize apostrophes
  $sent =~ s/([cdjlns])(')(.+?)/$1$2 $3/g;
  $sent =~ s/(qu')(.+?)/$1 $2/g;
  # dashes
  $sent =~ s/(\w)(\p{Dash_Punctuation}+)/$1 $2/g;
  my $bn = basename $j, ".wav";
  $p{$bn} = $sent;
}
close $P;

open my $W, '<', $w or croak "problem with $w $!";
open my $O, '+>', $o or croak "problem with $o $!";
open my $U, '+>', $u or croak "problem with $u $!";
open my $T, '+>', $t or croak "problem with $t $!";

LINE: while ( my $line = <$W> ) {
  chomp $line;
  my ($volume,$directories,$file) = File::Spec->splitpath( $line );
  my @dirs = split /\//, $directories;
  my $r = basename $line, ".wav";
  my $speaker = $dirs[-1];
  # only work with utterances in transcript file
  if ( exists $p{$r} ) {
    print $T "$r $p{$r}\n";
    print $O "$r sox $line -t .wav - |\n";
    print $U "$r $speaker\n";
  } else {
    warn "no transcript for $line";
  }
}
close $T;
close $O;
close $U;
close $W;
