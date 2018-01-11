#!/usr/bin/env perl

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
    open(I, "<$kwslist_in") || die "$0: Fail to open kwslist $kwslist_in\n";
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

sub KwslistOutputSort {
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
sub KwslistDupSort {
  my ($a, $b, $duptime) = @_;
  if ($a->[0] ne $b->[0]) {
    $a->[0] cmp $b->[0];
  } elsif ($a->[1] ne $b->[1]) {
    $a->[1] cmp $b->[1];
  } elsif ($a->[2] ne $b->[2]) {
    $a->[2] cmp $b->[2];
  } elsif (abs($a->[3]-$b->[3]) >= $duptime){
    $a->[3] <=> $b->[3];
  } elsif ($a->[5] ne $b->[5]) {
    $b->[5] <=> $a->[5];
  } else {
    $b->[4] <=> $a->[4];
  }
}

my $Usage = <<EOU;
This script reads a kwslist.xml file, does the post processing such as making decisions,
normalizing score, removing duplicates, etc. It writes the results to another kwslist.xml
file.

Usage: kwslist_post_process.pl [options] <kwslist_in|-> <kwslist_out|->
 e.g.: kwslist_post_process.pl kwslist.in.xml kwslist.out.xml

Allowed options:
  --beta        : Beta value when computing ATWV                (float,   default = 999.9)
  --digits      : How many digits should the score use          (int,     default = "infinite")
  --duptime     : Tolerance for duplicates                      (float,   default = 0.5)
  --duration    : Duration of the audio (Actural length/2)      (float,   default = 3600)
  --normalize   : Normalize scores or not                       (boolean, default = false)
  --Ntrue-scale : Keyword independent scale factor for Ntrue    (float,   default = 1.0)
  --remove-dup  : Remove duplicates                             (boolean, default = false)
  --remove-NO   : Remove the "NO" decision instances            (boolean, default = false)
  --verbose     : Verbose level (higher --> more kws section)   (integer, default 0)
  --YES-cutoff  : Only keep "\$YES-cutoff" yeses for each kw     (int,     default = -1)

EOU

my $beta = 999.9;
my $duration = 3600;
my $normalize = "false";
my $verbose = 0;
my $Ntrue_scale = 1.0;
my $remove_dup = "false";
my $duptime = 0.5;
my $remove_NO = "false";
my $digits = 0;
my $YES_cutoff = -1;
GetOptions('beta=f'     => \$beta,
  'duration=f'          => \$duration,
  'normalize=s'         => \$normalize,
  'verbose=i'           => \$verbose,
  'Ntrue-scale=f'       => \$Ntrue_scale,
  'remove-dup=s'        => \$remove_dup,
  'duptime=f'           => \$duptime,
  'remove-NO=s'         => \$remove_NO,
  'digits=i'            => \$digits,
  'YES-cutoff=i'        => \$YES_cutoff);

($normalize eq "true" || $normalize eq "false") || die "$0: Bad value for option --normalize\n";
($remove_dup eq "true" || $remove_dup eq "false") || die "$0: Bad value for option --remove-dup\n";
($remove_NO eq "true" || $remove_NO eq "false") || die "$0: Bad value for option --remove-NO\n";

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

# Removing duplicates
if ($remove_dup eq "true") {
  my @tmp = sort {KwslistDupSort($a, $b, $duptime)} @{$KWS};
  my @KWS = ();
  push(@KWS, $tmp[0]);
  for (my $i = 1; $i < scalar(@tmp); $i ++) {
    my $prev = $KWS[-1];
    my $curr = $tmp[$i];
    if ((abs($prev->[3]-$curr->[3]) < $duptime ) &&
        ($prev->[2] eq $curr->[2]) &&
        ($prev->[1] eq $curr->[1]) &&
        ($prev->[0] eq $curr->[0])) {
      next;
    } else {
      push(@KWS, $curr);
    }
  }
  $KWS = \@KWS;
}

my $format_string = "%g";
if ($digits gt 0 ) {
  $format_string = "%." . $digits ."f";
}

# Making decisions...
my %YES_count;
foreach my $kwentry (@{$KWS}) {
  my $threshold = $threshold{$kwentry->[0]};
  if ($kwentry->[5] > $threshold) {
    $kwentry->[6] = "YES";
    if (defined($YES_count{$kwentry->[0]})) {
      $YES_count{$kwentry->[0]} ++;
    } else {
      $YES_count{$kwentry->[0]} = 1;
    }
  } else {
    $kwentry->[6] = "NO";
    if (!defined($YES_count{$kwentry->[0]})) {
      $YES_count{$kwentry->[0]} = 0;
    }
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
      $kwentry->[5] = sprintf($format_string, $numerator/$denominator);
    } else {
      $kwentry->[5] = sprintf($format_string, $kwentry->[5]);
    }
  } else {
    $kwentry->[5] = sprintf($format_string, $kwentry->[5]);
  }
}

# Sorting and printing
my @tmp = sort KwslistOutputSort @{$KWS};

# Process the YES-cutoff. Note that you don't need this for the normal cases where
# hits and false alarms are balanced
if ($YES_cutoff != -1) {
  my $count = 1;
  for (my $i = 1; $i < scalar(@tmp); $i ++) {
    if ($tmp[$i]->[0] ne $tmp[$i-1]->[0]) {
      $count = 1;
      next;
    }
    if ($YES_count{$tmp[$i]->[0]} > $YES_cutoff*2) {
      $tmp[$i]->[6] = "NO";
      $tmp[$i]->[5] = 0;
      next;
    }
    if (($count == $YES_cutoff) && ($tmp[$i]->[6] eq "YES")) {
      $tmp[$i]->[6] = "NO";
      $tmp[$i]->[5] = 0;
      next;
    }
    if ($tmp[$i]->[6] eq "YES") {
      $count ++;
    }
  }
}

# Process the remove-NO decision
if ($remove_NO eq "true") {
  my @KWS = @tmp;
  @tmp = ();
  for (my $i = 0; $i < scalar(@KWS); $i ++) {
    if ($KWS[$i]->[6] eq "YES") {
      push(@tmp, $KWS[$i]);
    }
  }
}

# Printing
my $kwslist = PrintKwslist($info, \@tmp);

if ($kwslist_out eq "-") {
  print $kwslist;
} else {
  open(O, ">$kwslist_out") || die "$0: Fail to open output file $kwslist_out\n";
  print O $kwslist;
  close(O);
}
