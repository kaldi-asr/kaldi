#!/usr/bin/env perl

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.
#

use strict;
use warnings;
use Getopt::Long;

sub KeywordSort {
  if ($a->[0] ne $b->[0]) {
    $b->[0] <=> $a->[0];
  } else {
    $b->[1] <=> $a->[1];
  }
}

my $Usage = <<EOU;
This script reads a alignment.csv file and computes the ATWV, OTWV, MTWV by
sweeping the threshold. It also computes the lattice recall. The duration of
the search collection is supposed to be provided. In the Babel case, the
duration should be half of the total audio duration.

The alignment.csv file is supposed to have the following fields for each line:
language,file,channel,termid,term,ref_bt,ref_et,sys_bt,sys_et,sys_score,
sys_decision,alignment

Usage: kws_oracle_threshold.pl [options] <alignment.csv>
 e.g.: kws_oracle_threshold.pl alignment.csv

Allowed options:
  --beta                      : Beta value when computing ATWV              (float,   default = 999.9)
  --duration                  : Duration of all audio, you must set this    (float,   default = 999.9)

EOU

my $beta = 999.9;
my $duration = 999.9;
GetOptions(
  'beta=f'         => \$beta,
  'duration=f'     => \$duration);

@ARGV == 1 || die $Usage;

# Works out the input/output source.
my $alignment_in = shift @ARGV;

# Hash alignment file. For each instance we store a 3-dimension vector:
# [score, ref, res]
# where "score" is the confidence of that instance, "ref" equals 0 means there's
# no reference at that place and 1 means there's corresponding reference, "res"
# 0 means the instance is not considered when scoring, 1 means it's a false
# alarm and 2 means it's a true hit.
open(A, "<$alignment_in") || die "$0: Fail to open alignment file: $alignment_in\n";
my %Ntrue;
my %keywords;
my %alignment;
my $lattice_miss = 0;
my $lattice_ref = 0;
my %keywords_lattice_miss;
my %keywords_lattice_ref;
while (<A>) {
  chomp;
  my @col = split(',');
  @col == 12 || die "$0: Bad number of columns in $alignment_in: $_\n";

  # First line of the csv file.
  if ($col[11] eq "alignment") {next;}

  # Instances that do not have corresponding references.
  if ($col[11] eq "CORR!DET" || $col[11] eq "FA") {
    if (!defined($alignment{$col[3]})) {
      $alignment{$col[3]} = [];
    }
    my $ref = 0;
    my $res = 0;
    if ($col[11] eq "FA") {
      $res = 1;
    }
    push(@{$alignment{$col[3]}}, [$col[9], $ref, $res]);
    next;
  }

  # Instances that have corresponding references.
  if ($col[11] eq "CORR" || $col[11] eq "MISS") {
    if (!defined($alignment{$col[3]})) {
      $alignment{$col[3]} = [];
      $Ntrue{$col[3]} = 0;
      $keywords_lattice_miss{$col[3]} = 0;
      $keywords_lattice_ref{$col[3]} = 0;
    }
    my $ref = 1;
    my $res = 0;
    if ($col[10] ne "") {
      if ($col[11] eq "CORR") {
        $res = 2;
      }
      push(@{$alignment{$col[3]}}, [$col[9], $ref, $res]);
    }
    $Ntrue{$col[3]} += 1;
    $keywords{$col[3]} = 1;

    # The following is for lattice recall and STWV.
    $lattice_ref ++;
    $keywords_lattice_ref{$col[3]} ++;
    if ($col[11] eq "MISS" && $col[10] eq "") {
      $lattice_miss ++;
      $keywords_lattice_miss{$col[3]} ++;
    }
    next;
  }
}
close(A);

# Works out the oracle ATWV by sweeping the threshold.
my $atwv = 0.0;
my $otwv = 0.0;
my %mtwv_sweep;
foreach my $kwid (keys %keywords) {
  # Sort the instances by confidence score.
  my @instances = sort KeywordSort @{$alignment{$kwid}};
  my $local_otwv = 0.0;
  my $max_local_otwv = 0.0;
  my $local_atwv = 0.0;
  my $active_otwv_threshold = "";
  foreach my $instance (@instances) {
    my @ins = @{$instance};
    my $gain = 1.0 / $Ntrue{$kwid};
    my $cost = $beta / ($duration - $Ntrue{$kwid});
    # OTWV.
    if ($local_otwv > $max_local_otwv &&
        $active_otwv_threshold ne "" && $active_otwv_threshold != $ins[0]) {
      $max_local_otwv = $local_otwv;
    }
    if ($ins[1] == 1) {
      $local_otwv += $gain;
    } else {
      $local_otwv -= $cost;
    }
    $active_otwv_threshold = $ins[0];
    if ($active_otwv_threshold == 1.0) {
      # If score = 1.0, we always accept the instance as YES.
      $max_local_otwv = $local_otwv;
    }

    # ATWV.
    if ($ins[2] == 1) {
      $local_atwv -= $cost;
    } elsif ($ins[2] == 2) {
      $local_atwv += $gain;
    }

    # MTWV.
    for (my $threshold = 0.000; $threshold <= $ins[0]; $threshold += 0.001) {
      if ($ins[1] == 1) {
        $mtwv_sweep{$threshold} += $gain;
      } else {
        $mtwv_sweep{$threshold} -= $cost;
      }
    }
  }
  if ($local_otwv > $max_local_otwv) {
    $max_local_otwv = $local_otwv;
  }
  $atwv += $local_atwv;
  $otwv += $max_local_otwv;
}

# Works out the MTWV.
my $mtwv = 0.0;
my $mtwv_threshold = 0.0;
for my $threshold (keys %mtwv_sweep) {
  if ($mtwv_sweep{$threshold} > $mtwv) {
    $mtwv = $mtwv_sweep{$threshold};
    $mtwv_threshold = $threshold;
  }
}

# Works out the STWV.
my $stwv = 0.0;
for my $kw (keys %keywords_lattice_miss) {
  $stwv += $keywords_lattice_miss{$kw} / $keywords_lattice_ref{$kw};
}
$stwv = 1 - $stwv / scalar(keys %keywords);

$atwv /= scalar(keys %keywords);
$atwv = sprintf("%.4f", $atwv);
$otwv /= scalar(keys %keywords);
$otwv = sprintf("%.4f", $otwv);
$mtwv /= scalar(keys %keywords);
$mtwv = sprintf("%.4f", $mtwv);
my $lattice_recall = 1 - $lattice_miss / $lattice_ref;
$lattice_recall = sprintf("%.4f", $lattice_recall);
$stwv = sprintf("%.4f", $stwv);
print "ATWV = $atwv\n";
print "OTWV = $otwv\n";
print "STWV = $stwv\n";
print "MTWV = $mtwv, THRESHOLD = $mtwv_threshold\n";
print "Lattice Recall = $lattice_recall\n";
