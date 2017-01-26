#!/usr/bin/env perl
#===============================================================================
# Copyright 2016  (Author: Yenda Trmal <jtrmal@gmail.com>)
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
This script combines multiple KWS results files into one.

Usage: $0 [options] w1 <kwslist1> w2 <kwslist2> ... <kwslist_comb|->
 e.g.: $0 0.5 kwslist1.xml 0.5 kwslist2.xml ... kwslist_comb.xml

Allowed options:
  --probs           : The input scores are probabilities, not negative log-likelihoods)
  --method          : Use different combination method          (int,    default = 0)
                      0 -- CombSUM
                      1 -- CombMNZ
  --input-norm      : how the input data should be normalized   (int                  )
                      0 -- Saturate
                      1 -- NormSTO
                      2 -- source-wise NormSTO
  --output-norm    : how the output data should be normalized   (int                  )
                      0 -- Saturate
                      1 -- NormSTO
  --power           : The weighted power mean p-coefficient     (float,  default = 0.5)
  --gamma           : The gamma coefficient for CombMNZ         (float,  default = 0.0)
  --tolerance       : Tolerance (in frames) for being the same hits  (float,  default = 50)

EOU

use strict;
use warnings "FATAL";
use utf8;
use POSIX;
use Data::Dumper;
use Getopt::Long;
use File::Basename;
use Scalar::Util qw(looks_like_number);

$Data::Dumper::Indent = 2;

my $TOL = 50;
my $LIKES = 0;

sub OpenResults {
  my $list = shift @_;

  my $source = "STDIN";
  if ($list ne "-") {
    open(my $i, "<$list") || die "Fail to open file $list.\n";
    return  $i;
  }
  return $source
}

sub PrintResults {
  my $KWS = shift @_;

  # Start printing
  my $result = "";
  foreach my $kwentry (@{$KWS}) {
    my ($kwid, $file, $tbeg, $tend, $score, $dummy) = @{$kwentry};
    if ($score > 0) {
      $score = -log($score);
    } elsif ($score == 0) {
      $score = 9999;
    } else {
      die "Cannot take logarithm of a negative number\n" . join(" ", @{$kwentry}) . "\n";
    }
    $result .= "$kwid $file $tbeg $tend $score\n";
  }

  return $result;
}

sub KwslistTimeCompare {
  my ($a, $b) = @_;

  if ($a->[0] eq $b->[0]) { # KWID
    if ($a->[1] eq $b->[1]) { # FILEID
      if (abs($a->[2] - $b->[2]) <= $TOL) { # KW START
        if (abs($a->[3] - $b->[3]) <= $TOL) { #KW END
          return 0;
        } else {
          return ($a->[3] <=> $b->[3] );
        }
      } else {
        return $a->[2] <=> $b->[2];
      }
    } else {
      return $a->[1] cmp $b->[1];
    }
  } else {
    $a->[0] cmp $b->[0];
  }
}

sub KwslistTimeSort {
  my $a = shift;
  my $b = shift;
  return KwslistTimeCompare($a, $b);
}

sub ReadLines {
  my $kwid = shift @_;
  my %files = %{shift @_};
  my @lines = ();

  foreach my $id (sort keys %files) {
    my $l = readline $files{$id};
    next unless $l;
    chomp $l;
    my @entries = split " ", $l;
    while ($kwid eq $entries[0]) {
      push @entries, $id;
      push @lines, [@entries];

      $l = readline $files{$id};
      last unless $l;
      chomp $l;
      @entries = split " ", $l;
    }
    next unless defined $l;
    push @entries, $id;
    push @lines, [@entries];
  }
  return @lines;
}

sub ReadFirstLines {
  my %files = %{shift @_};
  my @lines = ();

  foreach my $id (sort keys %files) {
    my $l = readline $files{$id};
    next unless $l;
    chomp $l;

    my @entries = split " ", $l;
    push @entries, $id;
    push @lines, [@entries];
  }
  return @lines;
}

sub MergeCombPwrSum {
  my @results = @{shift @_};
  my %weights = %{shift @_};
  my $pwr = shift @_;
  my @output = ();

  return @output if not @results;

  while (@results) {
    my @mergelist = ();
    push @mergelist, shift @results;
    while ((@results) && (KwslistTimeCompare($mergelist[0], $results[0]) == 0)) {
      push @mergelist, shift @results;
    }

    my $best_score = -9999;
    my $tend;
    my $tbegin;
    my $out_score = 0;
    foreach my $elem (@mergelist) {
      my $score = $elem->[4];
      my $id = $elem->[5];
      if ($score > $best_score) {
        $best_score = $score;
        $tend = $elem->[3];
        $tbegin = $elem->[2];
      }
      #print "$out_score += $weights{$id} * $score\n";
      $out_score += $weights{$id} * ($score ** $pwr);
    }
    $out_score = $out_score**(1.0/$pwr);
    #print "$out_score \n\n\n";
    my $KWID = $mergelist[0]->[0];
    my $UTT = $mergelist[0]->[1];
    push @output, [$KWID, $UTT, $tbegin, $tend, $out_score, ""];
  }

  return \@output;
}

## More generic version of the combMNZ method
sub MergeCombPwrMNZ {
  my @results = @{shift @_};
  my %weights = %{shift @_};
  my $pwr = shift @_;
  my $gamma = shift @_;
  my @output = ();

  $gamma = 0 unless defined $gamma;
  return @output if not @results;

  while (@results) {
    my @mergelist = ();
    push @mergelist, shift @results;
    while ((@results) && (KwslistTimeCompare($mergelist[0], $results[0]) == 0)) {
      push @mergelist, shift @results;
    }

    my $best_score = -9999;
    my $tend;
    my $tbegin;
    my $out_score = 0;
    foreach my $elem (@mergelist) {
      my $score = $elem->[4];
      my $id = $elem->[5];
      if ($score > $best_score) {
        $best_score = $score;
        $tend = $elem->[3];
        $tbegin = $elem->[2];
      }
      #print "$out_score += $weights{$id} * $score\n";
      $out_score += $weights{$id} * ($score ** $pwr);
    }
    $out_score = (@mergelist ** $gamma) * $out_score**(1.0/$pwr);
    #print "$out_score \n\n\n";
    my $KWID = $mergelist[0]->[0];
    my $UTT = $mergelist[0]->[1];
    push @output, [$KWID, $UTT, $tbegin, $tend, $out_score, "out"];
  }

  return \@output;
}

### Sum-to-one normalization
sub NormalizeSTO {
  my @results = @{shift @_};
  my @output = ();
  my $sum = 0;
  foreach my $elem(@results) {
    $sum += $elem->[4];
  }
  foreach my $elem(@results) {
    $elem->[4] = $elem->[4]/$sum;
    push @output, $elem;
  }
  return \@output;
}

### This will STO normalize all entries in the @results according
### to the id, so that entries with the same id will sum to one
sub NormalizeSTOMulti {
  my @results = @{shift @_};
  my @output = ();
  my $sum = 0;
  my %sums = ();
  foreach my $elem(@results) {
    $sums{$elem->[5]} += $elem->[4];
  }
  foreach my $elem(@results) {
    $elem->[4] = $elem->[4]/$sums{$elem->[5]};
    push @output, $elem;
  }
  return \@output;
}

### Simple normalization of probabilities/scores
### Everything larger than 1 will be set to 1
sub NormalizeSaturate {
  my @results = @{shift @_};
  my @output = ();
  my $sum = 0;
  foreach my $elem(@results) {
    $elem->[4] = $elem->[4] > 1.0 ? 1.0 : $elem->[4];
    push @output, $elem;
  }
  return \@output;
}

my $method = 1;
my $input_norm = 0;
my $output_norm = 0;
my $gamma = 0;
my $power = 0.5;
GetOptions('tolerance=f'    => \$TOL,
           'method=i'       => sub { shift; $method = shift;
                                     if (($method lt 0) || ($method gt 1)) {
                                       die "Unknown method $method\n\n$Usage\n";
                                     }
                                   },
           'input-norm=i'       => sub { shift; my $n = shift;
                                     $input_norm = $n;
                                     if (($n lt 0) || ($n gt 2)) {
                                       die "Unknown input-norm $n\n\n$Usage\n";
                                     }
                                   },
           'output-norm=i'       => sub { shift; my $n = shift;
                                     $output_norm = $n;
                                     if (($n ne 0) || ($n ne 1)) {
                                       die "Unknown output-norm $n\n\n$Usage\n";
                                     }
                                   },
           'power=f'        => \$power,
           'gamma=f'        => \$gamma,
           'inv-power=f'    => sub {
                                    shift; my $val = shift;
                                    $power = 1.0/$val;
                                  },
           'probs'          => sub {
                                    $LIKES = 0;
                                  }
  ) || do {
  print STDERR "Cannot parse the command-line parameters.\n";
  print STDERR "$Usage\n";
  die "Cannot continue\n"
};

if (@ARGV % 2 != 1) {
  print STDERR "Bad number of (weight, results_list) pairs.\n";
  print STDERR "$Usage\n";
  die "Cannot continue\n"
}

# Workout the input/output source
my %results_files = ();
my %results_w = ();

my $i = 0;
while (@ARGV != 1) {
  my $w = shift @ARGV;
  looks_like_number($w) || die "$0: Bad weight: $w.\n";
  $results_w{$i} =  $w;
  $results_files{$i} = OpenResults(shift @ARGV);
  $i += 1;
}

my $sumw=0;
foreach my $val (values %results_w ) {
  $sumw += $val;
}
#foreach my $val (keys %results_w ) {
#  $results_w{$val} = $results_w{$val}/$sumw;
#}

my $output = shift @ARGV;

my $deb = 0;
my @lines = ();
@lines = ReadFirstLines(\%results_files);
@lines = sort { KwslistTimeSort($a, $b) } @lines;
push @lines, ReadLines($lines[0]->[0], \%results_files);
@lines = sort { KwslistTimeSort($a, $b) } @lines;

while (@lines) {
  my @res = ();

  push @res, shift @lines;
  while ((@lines) && ($lines[0]->[0] eq $res[0]->[0])) {
    push @res, shift @lines;
  }
  #print PrintResults(\@res);
  #print PrintResults(NormalizeSTO(MergeCombMNZ(\@res, \%results_w)));
  #print PrintResults(NormalizeCutoff(MergeCombPwrSum(\@res, \%results_w, $power)));
  #print PrintResults(NormalizeSaturate(MergeCombPwrMNZ(\@res, \%results_w, $power, $gamma)));
  #print PrintResults(NormalizeSTO(MergeCombPwrMNZ(NormalizeSTO(\@res), \%results_w, $power, $gamma)));

  my $data = undef;
  if ($input_norm == 1) {
    $data = NormalizeSTO(\@res);
  } elsif ($input_norm == 2) {
    $data = NormalizeSTOMulti(\@res);
  } else {
    $data = NormalizeSaturate(\@res);
  }

  if ($method == 0) {
    $data = MergeCombPwrSum($data, \%results_w, $power);
  } else {
    $data = MergeCombPwrMNZ($data, \%results_w, $power, $gamma);
  }

  if ($output_norm == 1) {
    $data = NormalizeSTO($data);
  } else {
    $data = NormalizeSaturate($data);
  }

  print PrintResults($data);

  #exit if $deb > 3;
  #$deb += 1 if $deb;
  #if ($res[0]->[0] eq "KW305-02318") {
  #  $deb = 1;
  #  print Dumper("START", \@res, \@lines) if $deb;
  #}

  my @tmp = ();
  if (@lines) {
    @tmp = ReadLines($lines[0]->[0], \%results_files);
  } else {
    # this is probably not necessary -- ReadLines() call
    # will always read one line _past_ the current KW
    # so we always should have extra KW in the @lines
    @tmp = ReadFirstLines(\%results_files);
  }

  #print Dumper("TMP", \@tmp) if $deb;
  if (@tmp > 0) {
    #print Dumper("XXX", \@res, \@lines) if $deb;
    push @lines, @tmp;
    @lines = sort { KwslistTimeSort($a, $b) } @lines;
  }

  #print Dumper(\@res, \@lines) if $deb;

}
