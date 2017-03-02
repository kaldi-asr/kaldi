#!/usr/bin/env perl

# Copyright 2012  Johns Hopkins University
# Apache 2.0.
#
use strict;
use warnings;
use XML::Simple;
use Data::Dumper;

binmode STDOUT, ":utf8";

my %seen;
while (my $keyword = <STDIN>) {
  chomp $keyword;
  $seen{$keyword} = 1;
}


my $data = XMLin($ARGV[0], ForceArray => 1);

#print Dumper($data->{kw});
my @filtered_kws = ();

foreach my $kwentry (@{$data->{kw}}) {
  if (defined $seen{$kwentry->{kwid}}) {
    push @filtered_kws, $kwentry;
  }
}
$data->{kw} = \@filtered_kws;
my $xml = XMLout($data, RootName=> "kwlist", KeyAttr=>'');
print $xml;
exit 0
