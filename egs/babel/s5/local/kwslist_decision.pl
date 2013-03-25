#!/usr/bin/perl

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.
#

use strict;
use warnings;
use Getopt::Long;

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
  my (@info, $kwid, $tbeg, $dur, $file, $score, $channel);
  my ($kwlist_filename, $language, $system_id) = ("", "", "");
  while (<$source>) {
    chomp;

    if (/<kwslist/) {
      /language="(\S+)"/ && ($language = $1);
      /system_id="(\S+)"/ && ($system_id = $1);
      /kwlist_filename="(\S+)"/ && ($kwlist_filename = $1);
      @info = ($kwlist_filename, $language, $system_id);
      next;
    }

    if (/<detected_kwlist/) {
      ($kwid) = /kwid="(\S+)"/;
      next;
    }

    if (/<kw/) {
      ($dur) = /dur="(\S+)"/;
      ($file) = /file="(\S+)"/;
      ($tbeg) = /tbeg="(\S+)"/;
      ($score) = /score="(\S+)"/;
      ($channel) = /channel="(\S+)"/;
      push(@KWS, [$kwid, $file, $channel, $tbeg, $dur, $score, ""]);
    }
  }

  $kwslist_in eq "-" || close(I);

  return [\@info, \@KWS];
}

sub PrintKwslist {
  my ($info, $KWS) = @_;

  my $kwslist = "";

  # Start printing
  $kwslist .= "<kwslist kwlist_filename=\"$info->[0]\" language=\"$info->[1]\" system_id=\"$info->[2]\">\n";
  my $prev_kw = "";
  foreach my $kwentry (@{$KWS}) {
    if ($prev_kw ne $kwentry->[0]) {
      if ($prev_kw ne "") {$kwslist .= "  </detected_kwlist>\n";}
      $kwslist .= "  <detected_kwlist search_time=\"1\" kwid=\"$kwentry->[0]\" oov_count=\"0\">\n";
      $prev_kw = $kwentry->[0];
    }
    $kwslist .= "    <kw file=\"$kwentry->[1]\" channel=\"$kwentry->[2]\" tbeg=\"$kwentry->[3]\" dur=\"$kwentry->[4]\" score=\"$kwentry->[5]\" decision=\"$kwentry->[6]\"";
    if (defined($kwentry->[7])) {$kwslist .= " threshold=\"$kwentry->[7]\"";}
    if (defined($kwentry->[8])) {$kwslist .= " raw_score=\"$kwentry->[8]\"";}
    $kwslist .= "/>\n";
  }
  $kwslist .= "  </detected_kwlist>\n";
  $kwslist .= "</kwslist>\n";

  return $kwslist;
}

sub KwslistSort {
  if ($a->[0] ne $b->[0]) {
    if ($a->[0] =~ m/[0-9]+$/ and $b->[0] =~ m/[0-9]+$/) {
      ($a->[0] =~ /([0-9]*)$/)[0] <=> ($b->[0] =~ /([0-9]*)$/)[0]
    } else {
      $a->[0] cmp $b->[0];
    }
  } elsif ($a->[5] ne $b->[5]) {
    $b->[5] <=> $a->[5];
  } else {
    $a->[1] cmp $b->[1];
  }
}

my $Usage = <<EOU;
Usage: kwslist_decision.pl [options] <kwslist_in|-> <kwslist_out|->
 e.g.: kwslist_decision.pl kwslist.in.xml kwslist.out.xml

Allowed options:
  --beta        : Beta value when computing ATWV                (float, default = 999.9)
  --duration    : Duration of the audio (Actural length/2)      (float, default = 3600)
  --normalize   : Normalize scores or not                       (boolean, default = false)
  --Ntrue-scale : Keyword independent scale factor for Ntrue    (float, default = 1.0)
  --verbose     : Verbose level (higher --> more kws section)   (integer, default 0)

EOU

my $beta = 999.9;
my $duration = 3600;
my $normalize = "false";
my $verbose = 0;
my $Ntrue_scale = 1.0;
GetOptions('beta=f'     => \$beta,
  'duration=f'          => \$duration,
  'normalize=s'         => \$normalize,
  'verbose=i'           => \$verbose,
  'Ntrue-scale=f'       => \$Ntrue_scale);

if ($normalize ne "true" && $normalize ne "false") {
  die "Bad value for option --normalize. \n";
}

@ARGV == 2 || die $Usage;

# Workout the input/output source
my $kwslist_in = shift @ARGV;
my $kwslist_out = shift @ARGV;

my ($info, $KWS) = @{ReadKwslist($kwslist_in)};

# Work out the Ntrue
my %Ntrue;
foreach my $kwentry (@{$KWS}) {
  if (!defined($Ntrue{$kwentry->[0]})) {
    $Ntrue{$kwentry->[0]} = 0.0;
  }
  $Ntrue{$kwentry->[0]} += $kwentry->[5];
}

# Scale the Ntrue and work out the expected count based threshold
my %threshold;
foreach my $key (keys %Ntrue) {
  $Ntrue{$key} *= $Ntrue_scale;
  $threshold{$key} = $Ntrue{$key}/($duration/$beta+($beta-1)/$beta*$Ntrue{$key});
}

# Making decisions...
foreach my $kwentry (@{$KWS}) {
  my $threshold = $threshold{$kwentry->[0]};
  if ($kwentry->[5] > $threshold) {
    $kwentry->[6] = "YES";
  } else {
    $kwentry->[6] = "NO";
  }
  if ($verbose > 0) {
    push(@{$kwentry}, sprintf("%g", $threshold));
  }
  if ($normalize eq "true") {
    if ($verbose > 0) {
      push(@{$kwentry}, $kwentry->[5]);
    }
    my $numerator = (1-$threshold)*$kwentry->[5];
    my $denominator = (1-$threshold)*$kwentry->[5]+(1-$kwentry->[5])*$threshold;
    if ($denominator != 0) {
      $kwentry->[5] = sprintf("%.2f", $numerator/$denominator);
    }
  }
}


# Sorting and printing
my @tmp = sort KwslistSort @{$KWS};
my $kwslist = PrintKwslist($info, \@tmp);

if ($kwslist_out eq "-") {
  print $kwslist;
} else {
  open(O, ">$kwslist_out") || die "Fail to open output file: $kwslist_out\n";
  print O $kwslist;
  close(O);
}
