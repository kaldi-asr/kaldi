#!/usr/bin/env perl
#===============================================================================
# Copyright 2015  (Author: Yenda Trmal <jtrmal@gmail.com>)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.
#===============================================================================
my $Usage = <<EOU;
Perform KST (keyword specific thresholt) normalization of the results

Usage: cat results | $0 [options]  > results.normalized

Allowed options:
  --probs               : the input is probabilities instead of neg-loglikelihoods

  --duration|--trials   : size of the searched collectiona in seconds (float)
  --beta                : the FA vs MISS rate    (float, default 999.9)
  --ntrue-scale         : scales for scaling the expected count of true hits (float, default 1.0)
  --thr|--threshold     : the decision threshold (float, default 0.5)
EOU

use strict;
use warnings;
use utf8;
use POSIX;
use Data::Dumper;
use Getopt::Long;

my $ntrue_scale = 1.0;
my $global_thr = 0.5;
my $beta = 999.9;
my $duration = 35785.578;
my $ntrue_table_filename;
my $probs=0;
my $bsum_filename;

GetOptions("duration|trials=f" => \$duration,
           "ntrue-scale=f"     => \$ntrue_scale,
           "beta=f"            => \$beta,
           "probs"             => \$probs,
           "thr|threshold=f"   => \$global_thr,
           "ntrue-table=s"     => \$ntrue_table_filename,
           "bsum-table=s"      => \$bsum_filename) or do
 {
  print STDERR "$0: Cannot parse the command-line parameters.\n";
  print STDERR "$Usage\n";
  die "$0: Cannot continue\n"
};

if (@ARGV != 0) {
  print STDERR "$0: Incorrect number of command-line parameters\n";
  print STDERR "$Usage\n";
  die "$0: Cannot continue\n"
}

sub ComputeKST {
  my @instances = @{shift @_};
  my $ntrue_scale = shift @_;
  my %ntrue_table = %{shift @_};


  my $ntrue = 0;
  foreach my $elem(@instances) {
    $ntrue += $elem->[4];
  }
  #$ntrue = $ntrue / @instances;
  if (defined ($ntrue_table{$instances[0]->[0]})) {
    #print STDERR "For KW "  . $instances[0]->[0] . " using the value " .  $ntrue_table{$instances[0]->[0]}  . "\n";
    $ntrue = $ntrue * $ntrue_table{$instances[0]->[0]};
  } else {
    #print STDERR  "Using the default vsalue $ntrue_scale\n";
    $ntrue = $ntrue * $ntrue_scale;
  }

  my $thr = $beta * $ntrue / ( $duration  + $ntrue * ($beta - 1));
  return $thr;
}

sub ComputeKSTWithExpected {
  my @instances = @{shift @_};
  my %expected_table = %{shift @_};
  my $ntrue_scale = shift @_;
  my %ntrue_table = %{shift @_};


  my $ntrue = $expected_table{$instances[0]->[0]};
  #$ntrue = $ntrue / @instances;
  if (defined ($ntrue_table{$instances[0]->[0]})) {
    #print STDERR "For KW "  . $instances[0]->[0] . " using the value " .  $ntrue_table{$instances[0]->[0]}  . "\n";
    $ntrue = $ntrue * $ntrue_table{$instances[0]->[0]};
  } else {
    #print STDERR  "Using the default vsalue $ntrue_scale\n";
    $ntrue = $ntrue * $ntrue_scale;
  }

  my $thr = $beta * $ntrue / ( $duration  + $ntrue * ($beta - 1));
  return $thr;
}
sub NormalizeScores {
  my @instances = @{shift @_};
  my $thr = shift @_;
  my $global_thr = shift @_;


  if ($thr == 0) {
    $thr = 0.001;
  }
  my $q = log($global_thr)/log($thr);

  foreach my $elem(@instances) {
    $elem->[4] = pow($elem->[4], $q);
  }
}

sub WriteResults {
  my @instances = @{shift @_};

  foreach my $elem(@instances) {
    print join(" ", @{$elem}) . "\n";
    die "$0: " . join(" ", @{$elem}) . "\n" if $elem->[-1] > 1.0;
  }

}

my $KWID;
my @putative_hits;
my %NTRUE_TABLE = ();

my %BSUM=();
if (defined $bsum_filename) {
  open(BSUMF, $bsum_filename) or die "$0: Cannot open $bsum_filename";
  while (my $line = <BSUMF> ) {
    chomp $line;
    next unless (($line =~ m/^\s*KW/) || ($line =~ m/^Keyword\s*KW/));
    $line =~ s/^Keyword//g;
    $line =~ s/^\s+|\s+$//g;
    my @entries = split /\s*\|\s*/, $line;
    $BSUM{$entries[0]} = $entries[12];
  }
  close(BSUMF);
}

if ( defined $ntrue_table_filename) {
  open (F, $ntrue_table_filename) or die "$0: Cannot open the Ntrue-table file\n";
  while (my $line = <F>) {
    my @entries=split(" ", $line);

    die "$0: The Ntrue-table does not have expected format\n" if @entries != 2;
    $NTRUE_TABLE{$entries[0]} = $entries[1] + 0.0;
  }
  close (F);
}

while (my $line = <STDIN>) {
  chomp $line;
  (my $kwid, my $file, my $start, my $end, my $score) = split " ", $line;

  if ($KWID && ($kwid ne $KWID)) {

    my $thr = ComputeKST(\@putative_hits, $ntrue_scale, \%NTRUE_TABLE );
    if ((defined $BSUM{$KWID}) && (scalar @putative_hits > 100)) {
      print STDERR "$0: $KWID $thr $BSUM{$KWID} " .  log($thr)/log($global_thr) . "\n";
      my $old_thr = $thr;
      $thr = pow($BSUM{$KWID}, log($thr)/log($global_thr));
    }
    if ($thr < 0.9999 ) {
      NormalizeScores(\@putative_hits, $thr, $global_thr);
      WriteResults(\@putative_hits);
    }

    $KWID = $kwid;
    @putative_hits = ();
  } elsif ( not $KWID ) {
    $KWID = $kwid;
  }

  unless ($probs) {
    $score = exp(-$score);
  }
  push @putative_hits, [$kwid, $file, $start, $end, $score];
}

if ($KWID) {
  my $thr = ComputeKST(\@putative_hits, $ntrue_scale, \%NTRUE_TABLE );
  if ((defined $BSUM{$KWID}) && (scalar @putative_hits > 100)) {
    $thr = pow($BSUM{$KWID}, log($thr)/log($global_thr));
  }
  if ($thr < 0.9999 ) {
    NormalizeScores(\@putative_hits, $thr, $global_thr);
    WriteResults(\@putative_hits);
  }
}

