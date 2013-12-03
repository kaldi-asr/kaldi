#!/usr/bin/env perl

# Copyright 2012  Johns Hopkins University
# Apache 2.0.
#
use strict;
use warnings;
use XML::Simple;
use Data::Dumper;

binmode STDOUT, ":utf8";
my @keywords;
while (my $line = <STDIN>) {
  chomp $line;
  push @keywords, $line;
}

#print "Will retain ",  scalar(@keywords), " keywords\n";

my $data = XMLin($ARGV[0], ForceArray => 1);

#print Dumper($data->{kw});
my @filtered_kws = ();

foreach my $kwentry (@{$data->{kw}}) {
  if ($kwentry->{kwid} ~~ @keywords ) {
    push @filtered_kws, $kwentry;
  }
}
$data->{kw} = \@filtered_kws;
my $xml = XMLout($data, RootName=> "kwlist", KeyAttr=>'');
print $xml; 
exit 0
