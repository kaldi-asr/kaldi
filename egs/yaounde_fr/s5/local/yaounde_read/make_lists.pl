#!/usr/bin/env perl

# Copyright 2018 John Morgan
# Apache 2.0.

# make_lists.pl - write lists for acoustic model training
# writes files under data/local/tmp/yaounde/read/lists

use strict;
use warnings;
use Carp;

BEGIN {
    @ARGV == 1 or croak "USAGE: $0 <DATA_SRC_DIR>
Example:
$0 African_Accented_French";
}

use File::Spec;
use File::Copy;
use File::Basename;

my ($d) = @ARGV;

# Initialize variables
my $tmpdir = "data/local/tmp/yaounde_read";
my $transcripts_file = "$d/transcripts/train/yaounde/fn_text.txt";
# In the future this file will  be changed to:
#my $transcripts_file = "$d/transcripts/train/yaounde/read/transcripts.txt";
# input wav file list
my $w = "$tmpdir/wav_list.txt";
# output temporary wav.scp file
my $wav_scp = "$tmpdir/lists/wav.scp";
# output temporary utt2spk file
my $utt_to_spk = "$tmpdir/lists/utt2spk";
# output temporary text file
my $txt_out = "$tmpdir/lists/text";
# initialize hash for transcripts
my %transcript = ();
# done setting variables

# This script looks at 2 files.
# One containing text transcripts and another containing file names for .wav files.
# It associates a text transcript with a .wav file name.

system "mkdir -p $tmpdir/lists";
open my $P, '<', $transcripts_file or croak "problem with $transcripts_file $!";
# store transcripts in hash
LINEA: while ( my $line = <$P> ) {
    chomp $line;
    next LINEA if ( $line =~ /answers/ );
  my ($utt_path,$sent) = split /\s/, $line, 2;
  my ($volume,$directories,$file) = File::Spec->splitpath( $utt_path );
  my $bn = basename $file, ".wav";
  my ($machine,$spk,$utt) = split /\-/, $bn, 3;
  next LINEA unless $utt =~ /\d\d\d\d/;
  my $utt_id = 'yaounde-' . $spk . '-' . $utt;
  # dashes?
  $sent =~ s/(\w)(\p{dash_punctuation}+?)/$1 $2/g;
  # a dangling quote?
  $sent =~ s/ ' / /g;
  $transcript{$utt_id} = $sent;
}
close $P;

open my $W, '<', $w or croak "problem with $w $!";
open my $WAVSCP, '+>', $wav_scp or croak "problem with $wav_scp $!";
open my $UTTSPK, '+>', $utt_to_spk or croak "problem with $utt_to_spk $!";
open my $TXT, '+>', $txt_out or croak "problem with $txt_out $!";

LINE: while ( my $line = <$W> ) {
  chomp $line;
  my ($volume,$directories,$file) = File::Spec->splitpath( $line );
  my @dirs = split /\//, $directories;
  my $mode = $dirs[4];
  next LINE unless ( $mode == 'read' );
  my $base = basename $line, ".wav";
  my ($maquina,$spk,$utt) = split /\-/, $base, 3;
  next LINE unless $utt =` /\d\d\d\d/;
  my $utt_id = 'yaounde-' . $spk . '-' . $utt;
  my $speaker = 'yaounde' . '-' . $spk;
  if ( defined $transcript{$utt_id} ) {
    print $TXT "$utt_id $transcript{$utt_id}\n";
    print $WAVSCP "$utt_id sox $line -t .wav - |\n";
    print $UTTSPK "$utt_id $speaker\n";
  } else {
      croak "Problem with $utt_id and $line";
  }
}
close $TXT;
close $WAVSCP;
close $UTTSPK;
close $W;
