#!/usr/bin/env perl

# Copyright 2013  Guoguo Chen
#           2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0.
#
# This script distributes data onto different file systems by making symbolic
# links. It is supposed to use together with utils/create_split_dir.pl, which
# creates a "storage" directory that links to different file systems.
#
# If a sub-directory egs/storage does not exist, it does nothing. If it exists,
# then it selects pseudo-randomly a number from those available in egs/storage/*
# creates a link such as
#
#   egs/egs.3.4.ark -> storage/4/egs.3.4.ark
#
use strict;
use warnings;
use File::Basename;
use File::Spec;
use Getopt::Long;

sub GetGCD {
  my ($a, $b) = @_;
  while ($a != $b) {
    if ($a > $b) {
      $a = $a - $b;
    } else {
      $b = $b - $a;
    }
  }
  return $a;
}

my $Usage = <<EOU;
This script distributes data onto different file systems by making symbolic
links. It is supposed to use together with utils/create_split_dir.pl, which
creates a "storage" directory that links to different file systems.

If a sub-directory foo/storage does not exist, it does nothing. If it exists,
then it selects pseudo-randomly a number from those available in foo/storage/*
creates a link such as

  foo/egs.3.4.ark -> storage/4/egs.3.4.ark

Usage: utils/create_data_link.pl <data-archive1> [<data-archive2> ... ]
 e.g.: utils/create_data_link.pl foo/bar/egs.3.4.ark foo/bar/egs.3.5.ark
 (note: the dirname, e.g. foo/bar/, must be the same in all cases).

See also utils/remove_data_links.sh
EOU

GetOptions();

if (@ARGV == 0) {
  die $Usage;
}

my $example_fullpath = $ARGV[0];

# Check if the storage has been created. If so, do nothing.
my $dirname = dirname($example_fullpath);
if (! -d "$dirname/storage") {
  exit(0);
}

# Storage exists, create symbolic links in the next few steps.

# First, get a list of the available storage directories, and check if they are
# properly created.
opendir(my $dh, "$dirname/storage/") || die "$0: Fail to open $dirname/storage/\n";
my @storage_dirs = grep(/^[0-9]*$/, readdir($dh));
closedir($dh);
my $num_storage = scalar(@storage_dirs);
for (my $x = 1; $x <= $num_storage; $x++) {
  (-d "$dirname/storage/$x") || die "$0: $dirname/storage/$x does not exist\n";
}

# Second, get the coprime list.
my @coprimes;
for (my $n = 1; $n < $num_storage; $n++) {
  if (GetGCD($n, $num_storage) == 1) {
    push(@coprimes, $n);
  }
}

my $ret = 0;

foreach my $fullpath (@ARGV) {
  if ($dirname ne dirname($fullpath)) {
    die "Mismatch in directory names of arguments: $example_fullpath versus $fullpath";
  }

  # Finally, work out the directory index where we should put the data to.
  my $basename = basename($fullpath);
  my $filename_numbers = $basename;
  $filename_numbers =~ s/[^0-9]+/ /g;
  my @filename_numbers = split(" ", $filename_numbers);
  my $total = 0;
  my $index = 0;
  foreach my $x (@filename_numbers) {
    if ($index >= scalar(@coprimes)) {
      $index = 0;
    }
    $total += $x * $coprimes[$index];
    $index++;
  }
  my $dir_index = $total % $num_storage + 1;

  # Make the symbolic link.
  if (-e $fullpath) {
    unlink($fullpath);
  }
  if (symlink("storage/$dir_index/$basename", $fullpath) != 1) { # failure
    $ret = 1;  # will exit with error status.
  }
}

exit($ret);

## testing:
# rm -rf foo bar
# mkdir -p bar/{1,2,3,4}
# mkdir -p foo/storage
# for x in 1 2 3 4; do ln -s ../../bar/$x foo/storage/$x; done
# utils/create_data_link.pl utils/create_data_link.pl foo/1.3.ark  foo/2.3.ark
# ls -l foo
# total 0
# lrwxrwxrwx 1 dpovey fax 17 Sep  2 17:41 1.3.ark -> storage/3/1.3.ark
# lrwxrwxrwx 1 dpovey fax 17 Sep  2 17:41 2.3.ark -> storage/4/2.3.ark
# drwxr-xr-x 2 dpovey fax 38 Sep  2 17:40 storage
