#!/usr/bin/env perl

# Copyright 2018 John Morgan
# Apache 2.0.

# make_lists.pl - write lists for acoustic model training
# writes files under data/local/tmp/ca16read_train/lists

use strict;
use warnings;
use Carp;

BEGIN {
  @ARGV == 1 or croak "USAGE: $0 <DATA_DIR>
Example:
$0 African_Accented_French
";
}

use File::Spec;
use File::Copy;
use File::Basename;

my ($d) = @ARGV;
# Initialize variables
my $tmpdir = "data/local/tmp/ca16read_train";
my $p = "$d/transcripts/train/ca16_read/conditioned.txt";
# input wav file list
my $wav_list = "$tmpdir/wav_list.txt";

# output temporary wav.scp files
my $wav_scp = "$tmpdir/lists/wav.scp";

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
  my ($x,$d,$s,$y,$i) = split /\_/, $j, 5;
  my $bn = 'gabonread_' . $s . '_' . $i;
  # dashes?
  $sent =~ s/(\w)(\p{dash_punctuation}+?)/$1 $2/g;
  $p{$bn} = $sent;
}
close $P;

open my $WAVLIST, '<', $wav_list or croak "problem with $wav_list $!";
open my $WAVSCP, '+>', $wav_scp or croak "problem with $wav_scp $!";
open my $U, '+>', $u or croak "problem with $u $!";
open my $T, '+>', $t or croak "problem with $t $!";

LINE: while ( my $line = <$WAVLIST> ) {
  chomp $line;
  my ($volume,$directories,$file) = File::Spec->splitpath( $line );
  my @dirs = split /\//, $directories;
  my $r = basename $line, ".wav";
  my ($x,$d,$s,$y,$i) = split /\_/, $r, 5;
  my $speaker = $dirs[-1];
  my $bn = 'gabonread_' . $s . '_' . $i;

  # only work with utterances in transcript file
  if ( exists $p{$bn} ) {
    my $fn = $bn . ".wav";
    print $T "$bn $p{$bn}\n";
    print $WAVSCP "$bn sox $line -t .wav - |\n";
    print $U "$bn gabonread_${s}\n";
  } else {
    # warn "no transcript for $line";
  }
}
close $T;
close $WAVSCP;
close $U;
close $WAVLIST;
