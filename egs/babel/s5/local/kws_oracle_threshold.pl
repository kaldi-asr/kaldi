#!/usr/bin/perl

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
This script reads a alignment.csv file and computes the oracle ATWV based on the
oracle threshold. The duration of the search collection is supposed to be provided.
In the Babel case, the duration should be half of the total audio duration.

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

# Workout the input/output source.
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
    next;
  }
}
close(A);

# Work out the oracle ATWV by sweeping the threshold.
my $atwv = 0.0;
my $oracle_atwv = 0.0;
foreach my $kwid (keys %keywords) {
  # Sort the instances by confidence score.
  my @instances = sort KeywordSort @{$alignment{$kwid}};
  my $local_oracle_atwv = 0.0;
  my $max_local_oracle_atwv = 0.0;
  my $local_atwv = 0.0;
  foreach my $instance (@instances) {
    my @ins = @{$instance};
    # Oracle ATWV.
    if ($ins[1] == 1) {
      $local_oracle_atwv += 1.0 / $Ntrue{$kwid};
    } else {
      $local_oracle_atwv -= $beta / ($duration - $Ntrue{$kwid});
    }
    if ($local_oracle_atwv > $max_local_oracle_atwv) {
      $max_local_oracle_atwv = $local_oracle_atwv;
    }

    # Original ATWV.
    if ($ins[2] == 1) {
      $local_atwv -= $beta / ($duration - $Ntrue{$kwid});
    } elsif ($ins[2] == 2) {
      $local_atwv += 1.0 / $Ntrue{$kwid};
    }
  }
  $atwv += $local_atwv;
  $oracle_atwv += $max_local_oracle_atwv;
}
$atwv /= scalar(keys %keywords);
$atwv = sprintf("%.4f", $atwv);
$oracle_atwv /= scalar(keys %keywords);
$oracle_atwv = sprintf("%.4f", $oracle_atwv);
print "Original ATWV = $atwv\n";
print "Oracle ATWV = $oracle_atwv\n";
