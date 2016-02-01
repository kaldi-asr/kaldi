#!/usr/bin/env perl

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.
#

# this is the original Guoguo's combination tweaked so that it works
# with the plain results files instead of the kwslist.xml files
use strict;
use warnings;
use Getopt::Long;
use Data::Dumper;
use File::Basename;

my $tolerance = 0.50;
my $flen = 0.01;
my $likes = 0;

sub ReadKwslist {
  my $kwslist_in = shift @_;

  my $source = "STDIN";
  if ($kwslist_in ne "-") {
    open(I, "<$kwslist_in") || die "Fail to open kwslist $kwslist_in.\n";
    $source = "I";
  }

  # Read in the kwslist and parse it. Note that this is a naive parse -- I simply
  # assume that the kwslist is "properly" generated
  my @KWS;
  my ($kwid, $tbeg, $dur, $file, $score, $channel);
  $channel = 1;
  while (<$source>) {
    chomp;
    my @entries = split " ";
    die "Unknown format of the result file, - expected 5 entries" unless @entries == 5;
    ($kwid, $file, $tbeg, $dur, $score) = @entries;
    $tbeg = $tbeg * $flen;
    $dur = $dur * $flen;

    $dur = $dur - $tbeg;
    $score = exp(-$score) if $likes;

    push(@KWS, [$kwid, $file, $channel, $tbeg, $dur, $score, ""]);
  }

  $kwslist_in eq "-" || close(I);

  my @info;
  return [\@info, \@KWS];
}

sub PrintKwslist {
  my ($info, $KWS) = @_;

  my $kwslist = "";

  # Start printing
  foreach my $kwentry (@{$KWS}) {
    my ($kwid, $file, $tbeg, $dur, $score, $dummy) = @{$kwentry};
    $dur = $tbeg + $dur;
    $tbeg = int($tbeg/$flen + 0.5);
    $dur = int($dur/$flen + 0.5);
    $score = -log($score);
    $kwslist .= "$kwid $file $tbeg $dur $score\n";
  }

  return $kwslist;
}

sub KwslistTimeCompare {
  my ($a, $b) = @_;

  if ($a->[0] eq $b->[0]) {
    if ($a->[1] eq $b->[1]) {
      if (abs($a->[3]-$b->[3]) <= $tolerance) {
        if (abs($a->[3]+$a->[4]-$b->[3]-$b->[4]) <= $tolerance) {
          return 0;
        } else {
          return ($a->[3]+$a->[4]) <=> ($b->[3]+$b->[4]);
        }
      } else {
        return $a->[3] <=> $b->[3];
      }
    } else {
      return $a->[1] cmp $b->[1];
    }
  } else {
    $a->[0] cmp $b->[0];
  } 
}

sub KwslistTimeSort {
  return KwslistTimeCompare($a, $b);
}

my $Usage = <<EOU;
Usage: $0 [options] w1 <kwslist1> w2 <kwslist2> ... <kwslist_comb|->
 e.g.: $0 0.5 kwslist1.xml 0.5 kwslist2.xml ... kwslist_comb.xml

Allowed options:
  --method        : Use different combination method          (int,    default = 1)
                    1 -- Weighted sum
                    2 -- Weighted "powered"
  --power         : The power of method 2                        (float,  default = 0.5)
  --tolerance     : Tolerance in frames for being the same hits  (float,  default = 50)

EOU

my $method = 1;
my $power = 0.5;
GetOptions('tolerance=f'    => \$tolerance, 
           'method=i'       => sub { shift; $method = shift; 
                                     if (($method ne 1) || ($method ne 2)) {
                                       die "Unknown method\n\n$Usage\n";
                                     }
                                   },
           'power=f'        => \$power,
           'inv-power=f'    => sub { 
                                    shift; my $val = shift; 
                                    $power = 1.0/$val;
                                  },
           'probs'          => sub { 
                                    $likes = 0; 
                                  }
  ) || die "Cannot parse the program options";

@ARGV >= 3 || die $Usage;

# Workout the input/output source
@ARGV % 2 == 1 || die "Bad number of (weight, kwslist) pair.\n";
my @kwslist_file = ();
my @weight = ();
while (@ARGV != 1) {
  my $w = shift @ARGV;
  $w =~ m/^[0-9.]*$/ || die "Bad weight: $w.\n";
  push(@weight, $w);
  push(@kwslist_file, shift @ARGV);
}
my $output = shift @ARGV;

# Open the first kwslist
my ($info, $KWS) = @{ReadKwslist($kwslist_file[0])};

# Open the rest kwslists
my @kwslist = ();
for (my $i = 1; $i < @kwslist_file; $i ++) {
  push(@kwslist, @{ReadKwslist($kwslist_file[$i])}[1]);
}

# Process the first kwslist
my @KWS = sort KwslistTimeSort @{$KWS};
my $w = shift @weight;
foreach my $kwentry (@$KWS) {
  if ($method == 1) {
    $kwentry->[5] = $kwentry->[5] * $w;
  } elsif ($method == 2) {
    $kwentry->[5] = ($kwentry->[5]**$power) * $w;
  } else {
    die "Method not defined.\n";
  }
}

# Start merging the rest kwslists
while (@kwslist > 0) {
  my $w = shift @weight;
  my @kws = sort KwslistTimeSort @{shift @kwslist};

  # We'll take time information from the first system
  my ($i, $j) = (0, 0);
  my @from_kws;
  while ($i < @KWS and $j < @kws) {
    my $cmp = KwslistTimeCompare($KWS[$i], $kws[$j]);
    if ($cmp == 0) {
      if ($method == 1) {
        $KWS[$i]->[5] += $kws[$j]->[5] * $w;
      } elsif ($method == 2) {
        $KWS[$i]->[5] += ($kws[$j]->[5]**$power) * $w;
      } else {
        die "Method not defined.\n";
      }
      $i ++;
      $j ++;
    } elsif ($cmp == -1) {
      $i ++;
    } else {
      if ($method == 1) {
        $kws[$j]->[5] = $kws[$j]->[5] * $w;
      } elsif ($method == 2) {
        $kws[$j]->[5] = ($kws[$j]->[5]**$power) * $w;
      } else {
        die "Method not defined.\n";
      }
      push(@from_kws, $kws[$j]);
      $j ++;
    }
  }
  while ($j < @kws) {
    if ($method == 1) {
      $kws[$j]->[5] = $kws[$j]->[5] * $w;
    } elsif ($method == 2) {
      $kws[$j]->[5] = ($kws[$j]->[5]**$power) * $w;
    } else {
      die "Method not defined.\n";
    }
    push(@from_kws, $kws[$j]);
    $j ++;
  }

  # Sort again
  @from_kws = (@KWS, @from_kws);
  @KWS = sort KwslistTimeSort @from_kws;
}

if ($method == 2) {
  foreach my $kwentry (@KWS) {
    $kwentry->[5] = $kwentry->[5]**(1.0/$power);
    $kwentry->[5] = 1.0 if  $kwentry->[5] > 1.0;
  }
}

# Sorting and pringting
my $kwslist = PrintKwslist(\@{$info}, \@KWS);

if ($output eq "-") {
  print $kwslist;
} else {
  open(O, ">$output") || die "Fail to open output file: $output\n";
  print O $kwslist;
  close(O);
}
