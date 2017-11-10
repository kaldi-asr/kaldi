#!/usr/bin/env perl

# Copyright 2017 John Morgan
# Apache 2.0.

# heroico_answers_make_lists.pl - make acoustic model training lists

use strict;
use warnings;
use Carp;

use File::Spec;
use File::Copy;
use File::Basename;

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

# store prompts in hash
LINEA: while ( my $line = <> ) {
  chomp $line;
  my ($num,$sent) = split /\t/, $line, 2;
  my ($volume,$directories,$file) = File::Spec->splitpath( $num );
  my @dirs = split /\//, $directories;
  # get the speaker number
  my $s = $dirs[-1];
  # the utterance name
  my $i = $s . '_' . 'a' . '_' . $file;
  $p{$i} = $sent;
}

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
  my $rid = $s . '_' . 'a' . '_' . $r;
  if ( exists $p{$rid} ) {
    print $T "$rid $p{$rid}\n";
  } elsif ( defined $rid ) {
    warn  "problem\t$rid";
    next LINE;
  } else {
    croak "$line";
  }

  print $O "$rid sox -r 22050 -e signed -b 16 $line -r 16000 -t wav - |\n";
  print $U "$rid ${s}_a\n";
}
close $T;
close $O;
close $U;
close $W;
