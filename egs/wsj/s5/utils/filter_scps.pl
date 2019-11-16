#!/usr/bin/env perl
# Copyright 2010-2012   Microsoft Corporation
#           2012-2016   Johns Hopkins University (author: Daniel Povey)
#                2015   Xiaohui Zhang

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


$field = 1;
$shifted = 0;
$print_warnings = 1;
do {
  $shifted=0;
  if ($ARGV[0] eq "-f") {
    $field = $ARGV[1];
    shift @ARGV; shift @ARGV;
    $shifted = 1;
  }
  if (@ARGV[0] eq "--no-warn") {
    $print_warnings = 0;
    shift @ARGV;
    $shifted = 1;
  }
} while ($shifted);


if(@ARGV != 4) {
  die "Usage: utils/filter_scps.pl [-f <field-to-filter-on>] <job-range-specifier> <filter-pattern> <input-scp> <output-scp-pattern>\n" .
       "e.g.:  utils/filter_scps.pl  JOB=1:10 data/train/split10/JOB/spk2utt data/train/feats.scp data/train/split10/JOB/feats.scp\n" .
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

$idlist = shift @ARGV;

if ($idlist !~ m/$jobname/ &&
    $jobend > $jobstart) {
  print STDERR "filter_scps.pl: you are trying to use multiple filter files as filter patterns but "
    . "you are providing just one filter file ($idlist)\n";
  exit(1);
}


$infile = shift @ARGV;

$outfile = shift @ARGV;

if ($outfile !~ m/$jobname/ &&  $jobend > $jobstart) {
  print STDERR "filter_scps.pl: you are trying to create multiple filtered files but "
    . "you are providing just one output file ($outfile)\n";
  exit(1);
}

# This hashes from the id (e.g. utterance-id) to an array of the relevant
# job-ids (which are integers).  In any normal use-case, this array will contain
# exactly one job-id for any given id, but we want to be agnostic about this.
%id2jobs = ( );

# Some variables that we set to produce a warning.
$warn_uncovered = 0;
$warn_multiply_covered = 0;

for ($jobid = $jobstart; $jobid <= $jobend; $jobid++) {
  $idlist_n = $idlist;
  $idlist_n =~ s/$jobname/$jobid/g;

  open(F, "<$idlist_n") || die "Could not open id-list file $idlist_n";

  while(<F>) {
    @A = split;
    @A >= 1 || die "Invalid line $_ in id-list file $idlist_n";
    $id = $A[0];
    if (! defined $id2jobs{$id}) {
      $id2jobs{$id} = [ ];  # new anonymous array.
    }
    push @{$id2jobs{$id}}, $jobid;
  }
  close(F);
}

# job2output hashes from the job-id, to an anonymous array containing
# a sequence of output lines.
%job2output = ( );
for ($jobid = $jobstart; $jobid <= $jobend; $jobid++) {
  $job2output{$jobid} = [ ];  # new anonymous array.
}

open (F, "< $infile") or die "Can't open $infile for read: $!";
while (<F>) {
  if ($field == 1) {           # Treat this as special case, since it is common.
    $_ =~ m/\s*(\S+)\s*/ || die "Bad line $_, could not get first field.";
    # $1 is what we filter on.
    $id = $1;
  } else {
    @A = split;
    @A > 0 || die "Invalid scp file line $_";
    @A >= $field || die "Invalid scp file line $_";
    $id = $A[$field-1];
  }
  if ( ! defined $id2jobs{$id}) {
    $warn_uncovered = 1;
  } else {
    @jobs = @{$id2jobs{$id}};   # this dereferences the array reference.
    if (@jobs > 1) {
      $warn_multiply_covered = 1;
    }
    foreach $job_id (@jobs) {
      if (!defined $job2output{$job_id}) {
        die "Likely code error";
      }
      push @{$job2output{$job_id}}, $_;
    }
  }
}
close(F);

for ($jobid = $jobstart; $jobid <= $jobend; $jobid++) {
  $outfile_n = $outfile;
  $outfile_n =~ s/$jobname/$jobid/g;
  open(FW, ">$outfile_n") || die "Could not open output file $outfile_n";
  $printed = 0;
  foreach $line (@{$job2output{$jobid}}) {
    print FW $line;
    $printed = 1;
  }
  if (!printed) {
    print STDERR "filter_scps.pl: warning: output to $outfile_n is empty\n";
  }
  close(FW);
}

if ($warn_uncovered && $print_warnings) {
  print STDERR "filter_scps.pl: warning: some input lines did not get output\n";
}
if ($warn_multiply_covered && $print_warnings) {
  print STDERR "filter_scps.pl: warning: some input lines were output to multiple files [OK if splitting per utt] " .
    join(" ", @ARGV) . "\n";
}
