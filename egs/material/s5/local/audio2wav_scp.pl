#!/usr/bin/env perl
#===============================================================================
# Copyright 2017  (Author: Yenda Trmal <jtrmal@gmail.com>)
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

use strict;
use warnings;
use utf8;


my $sox =  `which sox` or die "The sox binary does not exist";
chomp $sox;
my $sph2pipe = `which sph2pipe` or die "The sph2pipe binary does not exist";
chomp $sph2pipe;

while(<STDIN>) {
  chomp;
  my $full_path = $_;
  (my $basename = $full_path) =~ s/.*\///g;

  die "The filename $basename does not match the expected naming pattern!" unless $basename =~ /.*\.(wav|sph)$/;
  (my $ext = $basename) =~ s/.*\.(wav|sph)$/$1/g;
  (my $name = $basename) =~ s/(.*)\.(wav|sph)$/$1/g;


  # name looks like this:
  #   MATERIAL_BASE-1A-BUILD_10002_20131130_011225_inLine.sph
  # Please note that the naming pattern must match
  # the pattern in create_datafiles.pl
  $name =~ s/inLine.*/0/g;
  $name =~ s/outLine.*/1/g;
  $name =~ s/_BASE//g;
  $name =~ s/-BUILD//g;

  if ($ext eq "wav") {
    print "$name $sox $full_path -r 8000 -c 1 -b 16 -t wav - downsample|\n";
  } else {
    print "$name $sph2pipe -f wav -p -c 1 $full_path|\n";
  }
}


