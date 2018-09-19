#!/usr/bin/env perl
# Input buckwalter encoded Arabic and print it out as utf-8 encoded Arabic.
use strict;
use warnings;
use Carp;

use Encode::Arabic::Buckwalter;         # imports just like 'use Encode' would, plus more

while ( my $line = <>) {
    print encode 'utf8', decode 'buckwalter', $line;
}
