#!/usr/bin/perl
# Copyright 2010-2012 Microsoft Corporation
#                     Johns Hopkins University (author: Daniel Povey)
#           2015      Xiaohui Zhang

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


# This script takes multiple lists of utterance-ids or any file whose first field
# of each line is an utterance-id, as filters, and filters an scp
# file (or any file whose "n-th" field is an utterance id), printing
# out only those lines whose "n-th" field is in filter. The index of
# the "n-th" field is 1, by default, but can be changed by using
# the -f <n> switch


if(@ARGV != 4) {
  die "Usage: utils/filter_scps.pl  <job-range-specifier> <filter-pattern> <input-scp> <output-scp-pattern>\n" .
       "e.g.:  utils/filter_scps.pl [-f <field-to-filter-on>] JOB=1:10 data/train/split10/JOB/spk2utt data/train/feats.scp data/train/split10/JOB/feats.scp\n" .
       "similar to utils/filter_scp.pl, but it uses multiple filters and output multiple filtered files.\n".
       "The -f option specifies the field in <input-scp> that we filter on (default: 1)." .
       "See also: utils/filter_scp.pl\n";
}

if ($ARGV[0] =~ m/^([\w_][\w\d_]*)+=(\d+):(\d+)$/) { # e.g. JOB=1:10
  $jobname = $1;
  $jobstart = $2;
  $jobend = $3;
  shift;
  if ($jobstart > $jobend) {
    die "filter_scps.pl: invalid job range $ARGV[0]";
  }
} else {
  die "filter_scps.pl: bad job-range specifier $ARGV[0]: expected e.g. JOB=1:10";
}

$field = 1;
$shifted = 0;
do {
  $shifted=0;
  if ($ARGV[0] eq "-f") {
    $field = $ARGV[1];
    shift @ARGV; shift @ARGV;
    $shifted=1
  }
} while ($shifted);

$idlist = shift @ARGV;

if (defined $jobname && $idlist !~ m/$jobname/ &&
    $jobend > $jobstart) {
  print STDERR "filter_scps.pl: you are trying to use multiple filter files as filter patterns but "
    . "you are providing just one filter file ($idlist)\n";
  exit(1);
}


$infile = shift @ARGV;
open (F, "< $infile") or die "Can't open $infile for read: $!";
my @inlines;
@inlines = <F>;
close(F);

$outfile = shift @ARGV;

if (defined $jobname && $outfile !~ m/$jobname/ &&
    $jobend > $jobstart) {
  print STDERR "filter_scps.pl: you are trying to create multiple filtered files but "
    . "you are providing just one output file ($outfile)\n";
  exit(1);
}

for ($jobid = $jobstart; $jobid <= $jobend; $jobid++) {
  $outfile_n = $outfile;
  $idlist_n = $idlist;
  if (defined $jobname) { 
    $idlist_n =~ s/$jobname/$jobid/g;
    $outfile_n =~ s/$jobname/$jobid/g;
  }

  open(F, "<$idlist_n") || die "Could not open id-list file $idlist_n";
  my %seen;
  while(<F>) {
    @A = split;
    @A>=1 || die "Invalid line $_ in id-list file $idlist_n";
    $seen{$A[0]} = 1;
  }
  close(F);
  open(FW, ">$outfile_n") || die "Could not open output file $outfile_n";
  foreach(@inlines) {
    if ($field == 1) { # Treat this as special case, since it is common.
      $_ =~ m/\s*(\S+)\s*/ || die "Bad line $_, could not get first field.";
      # $1 is what we filter on.
      if ($seen{$1}) {
        print FW $_;
      }
    } else {
      @A = split;
      @A > 0 || die "Invalid scp file line $_";
      @A >= $field || die "Invalid scp file line $_";
      if ($seen{$A[$field-1]}) {
        print FW $_;
      }
    }
  }
  close(FW);
}
