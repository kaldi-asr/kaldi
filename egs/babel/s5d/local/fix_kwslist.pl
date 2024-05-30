#!/usr/bin/env perl

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen, Jan Trmal)
# Apache 2.0.
#

use strict;
use warnings;
use Getopt::Long;
use XML::Simple;
use Data::Dumper;
use File::Basename;

sub mysort {
  if ($a->{kwid} =~ m/[0-9]+$/ and $b->{kwid} =~ m/[0-9]+$/) {
    ($a->{kwid} =~ /([0-9]*)$/)[0] <=> ($b->{kwid} =~ /([0-9]*)$/)[0]
  } else {
    $a->{kwid} cmp $b->{kwid};
  }
}

my $Usage = <<EOU;
Usage: fix_kwslist.pl [options] <kwlist_in> <kwslist_in|-> <fixed_kwslist_out|->
 e.g.: fix_kwslist.pl --kwlist-filename=kwlist.xml kwlist.xml kwslist.xml fixed_kwslist.xml

Allowed options:
  --kwlist-filename       : Kwlist filename with version info     (string, default = "")

EOU

my $kwlist_filename="";
GetOptions('kwlist-filename=s'    => \$kwlist_filename);

if (@ARGV != 3) {
  die $Usage;
}

# Workout the input/output source
my $kwlist_in = shift @ARGV;
my $kwslist_in = shift @ARGV;
my $fixed_kwslist_out = shift @ARGV;

my $KW = XMLin($kwlist_in);
my $KWS = XMLin($kwslist_in);

# Extract keywords from kwlist.xml
my %kwlist;
my $language = $KW->{language};
foreach my $kwentry (@{$KW->{kw}}) {
  $kwlist{$kwentry->{kwid}} = 1;
}

# Now work on the kwslist
$KWS->{language} = $language;
if ($kwlist_filename ne "") {
  $KWS->{kwlist_filename} = basename($kwlist_filename);
} elsif ($KWS->{kwlist_filename} eq "") {
  $KWS->{kwlist_filename} = basename($kwlist_in);
}
foreach my $kwentry (@{$KWS->{detected_kwlist}}) {
  if (defined($kwlist{$kwentry->{kwid}})) {
    delete $kwlist{$kwentry->{kwid}};
  }
}

# Empty entries...
foreach my $kw (keys %kwlist) {
  my %empty;
  my @tmp = [];
  $empty{search_time} = 1;
  $empty{kwid} = $kw;
  $empty{oov_count} = 0;
  push(@{$KWS->{detected_kwlist}}, \%empty);
}

my @sorted = sort mysort @{$KWS->{detected_kwlist}};
$KWS->{detected_kwlist} = \@sorted;

my $xml = XMLout($KWS, RootName => "kwslist", NoSort=>0);
if ($fixed_kwslist_out eq "-") {
  print $xml;
} else {
  if (!open(O, ">$fixed_kwslist_out")) {
    print "Fail to open output file: $fixed_kwslist_out\n";
    exit 1;
  }
  print O $xml;
  close(O);
}
