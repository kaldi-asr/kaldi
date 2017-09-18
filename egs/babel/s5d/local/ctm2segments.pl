#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Long;

my $cf_needed = 0.9;
my $cf_needed_upper = 1;
my $extend_segments = 0.0 ;

my $Usage = <<EOU;
Generate kaldi text and segments file (for use during unsupervised training)
Usage:    ctm_to_training.pl [options] <ctm_in|-> <text_out|->

Allowed options:
  --min-cf           : Minimum CF to include the word (float, default = 0.9)
  --max-cf           : Maximum CF to include the word (float, default = 1.0)
  --extend-segments  : Add this delta to the boundaries of the segments (float, default = 0.0)
EOU

GetOptions('min-cf=f'   => \$cf_needed,
           'max-cf=f'   => \$cf_needed_upper,
           'extend-segments=f'  => \$extend_segments,
           );


# Get parameters
my $filein = shift @ARGV;
my $dirout = shift @ARGV;


my @segments;
my @utterances;
my @text;

my $words = "";
my $seg_end = -1;
my $seg_start = -1;
my $filename;

my $total_seconds=0;
my $extracted_seconds=0;
open(FILEIN, $filein);
while (my $line=<FILEIN> ) {
  chop $line;
  my @entries = split(/ /, $line);
  die "Cannot parse line \"$line\""  if scalar @entries != 6;

  ($filename, my $chann_id, my $beg, my $end, my $word, my $conf) = @entries;

  $total_seconds += $end * 1.0;

  if ($conf >= $cf_needed ) {
    if ( $words ne "" ) {
      #print "Extend segment\n";
      $words .= " $word";
      $seg_end = $beg * 1.0 + $end*1.0;
    } else {
      #start a new segment
      #print "Start segment\n";
      $seg_start = $beg;
      $seg_end = $beg * 1.0 + $end*1.0;
      $words = $word;
    }
  } else {
    #flush the segment
    if ( $words ) {
      my @filename_parts = split(/_/, $filename);
      my $channel="C";
      if ($filename_parts[6] eq "inLine" ) {
        $channel="A";
      } elsif ($filename_parts[6] eq "outLine" ) {
        $channel="B";
      }

      $extracted_seconds+= ($seg_end - $seg_start);
      $seg_start -= $extend_segments;
      $seg_end += $extend_segments;

      my $spk_id=$filename_parts[3] . "_" . $channel;
      my $utt_id = $spk_id . "_" . join("_", @filename_parts[4..5]);
      my $last_part = sprintf("%06d", $seg_start * 100);
      $utt_id .= "_" . $last_part;
      #print $utt_id . "  $beg \n";

      #14350_A_20121123_042710_001337

      #10901_A_20121128_230024_000227 BABEL_OP1_206_10901_20121128_230024_inLine 2.275 3.265
      my $segment = "$utt_id $filename $seg_start $seg_end";
      #14350_A_20121123_042710_001337 14350_A
      my $utt2spk = "$utt_id $spk_id";
      #10901_A_20121128_230024_000227 hayi Lovemore
      my $text = "$utt_id $words";
      push @segments, $segment;
      push @utterances, $utt2spk;
      push @text, $text;
      $words = "";
    }

  }
}
if ( $words ) {
  #print "Flush.\n";
  my @filename_parts = split(/_/, $filename);
  my $channel="C";
  if ($filename_parts[6] eq "inLine" ) {
    $channel="A";
  } elsif ($filename_parts[6] eq "outLine" ) {
    $channel="B";
  }

  $extracted_seconds+= ($seg_end - $seg_start);
  $seg_start -= $extend_segments;
  $seg_end += $extend_segments;

  my $spk_id=$filename_parts[3] . "_" . $channel;
  my $utt_id = $spk_id . "_" . join("_", @filename_parts[4..5]);
  my $last_part = sprintf("%06d", $seg_start * 100);
  $utt_id .= "_" . $last_part;
  #print $utt_id . "  $beg \n";

  #14350_A_20121123_042710_001337

  #10901_A_20121128_230024_000227 BABEL_OP1_206_10901_20121128_230024_inLine 2.275 3.265
  my $segment = "$utt_id $filename $seg_start $seg_end";
  #14350_A_20121123_042710_001337 14350_A
  my $utt2spk = "$utt_id $spk_id";
  #10901_A_20121128_230024_000227 hayi Lovemore
  my $text = "$utt_id $words";
  push @segments, $segment;
  push @utterances, $utt2spk;
  push @text, $text;
  $words = "";
}

open(SEGMENTS, "> $dirout/segments");
foreach my $line (@segments) {
  print SEGMENTS "$line\n";
}
close(SEGMENTS);

open(TEXT, "> $dirout/text");
foreach my $line (@text) {
  print TEXT "$line\n";
}
close(TEXT);

open(UTT, "> $dirout/utt2spk");
foreach my $line (@utterances) {
  print UTT "$line\n";
}
close(UTT);

my $total_hours=sprintf("%.2f", $total_seconds/3600);
my $extracted_hours=sprintf("%.2f", $extracted_seconds/3600);
my $s_ex_secs=sprintf("%d", $extracted_seconds);

print "Fragments extracted: $s_ex_secs seconds ($extracted_hours hours) out of $total_hours hours\n";

