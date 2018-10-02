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
my %prompts = ();

# store prompts in hash
LINEA: while ( my $line = <> ) {
  chomp $line;
  my ($num,$sent) = split /\t/, $line, 2;
  my ($volume,$directories,$file) = File::Spec->splitpath( $num );
  my @dirs = split /\//, $directories;
  # get the speaker number
  my $s = $dirs[-1];
  # pad the speaker number with zeroes
  my $spk = "";
  if ( $s < 10 ) {
      $spk = '000' . $s;
  } elsif ( $s < 100 ) {
      $spk = '00' . $s;
  } elsif ( $s < 1000 ) {
      $spk = '0' . $s;
  }
  # pad the filename with zeroes
  my $fn = "";
  if ( $file < 10 ) {
      $fn = '000' . $file;
  } elsif ( $file < 100 ) {
      $fn = '00' . $file;
  } elsif ( $file < 1000 ) {
      $fn = '0' . $file;
  }
  # the utterance name
  my $utt = $spk . '_' . $fn;
  $prompts{$utt} = $sent;
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
  my $spk = "";
  # pad with zeroes
  if ( $s < 10 ) {
      $spk = '000' . $s;
  } elsif ( $s < 100 ) {
      $spk = '00' . $s;
  } elsif ( $s < 1000 ) {
      $spk = '0' . $s;
  }
  # pad the file name with zeroes
  my $rec = "";
  if ( $r < 10 ) {
      $rec = '000' . $r;
  } elsif ( $r < 100 ) {
      $rec = '00' . $r;
  } elsif ( $r < 1000 ) {
      $rec = '0' . $r;
  }
  my $rec_id = $spk . '_' . $rec;
  if ( exists $prompts{$rec_id} ) {
    print $T "$rec_id $prompts{$rec_id}\n";
  } elsif ( defined $rec_id ) {
    warn  "warning: problem\t$rec_id";
    next LINE;
  } else {
    croak "$line";
  }

  print $O "$rec_id sox -r 22050 -e signed -b 16 $line -r 16000 -t wav - |\n";
  print $U "$rec_id $spk\n";
}
close $T;
close $O;
close $U;
close $W;
