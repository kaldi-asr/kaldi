#!/usr/bin/env perl

# Copyright 2013  Guoguo Chen
# Apache 2.0.
#
# This script creates storage directories on different file systems, and creates
# symbolic links to those directories. For example, a command
#
#   utils/create_split_dir.pl /export/gpu-0{3,4,5}/egs/storage egs/storage
#
# will mkdir -p all of those directories, and will create links
#
#   egs/storage/1 -> /export/gpu-03/egs/storage
#   egs/storage/2 -> /export/gpu-03/egs/storage
#   ...
#
use strict;
use warnings;
use File::Spec;
use Getopt::Long;

my $Usage = <<EOU;
create_split_dir.pl:
This script creates storage directories on different file systems, and creates
symbolic links to those directories.

Usage: utils/create_split_dir.pl <actual_storage_dirs> <pseudo_storage_dir>
 e.g.: utils/create_split_dir.pl /export/gpu-0{3,4,5}/egs/storage egs/storage

Allowed options:
  --suffix    : Common suffix to <actual_storage_dirs>    (string, default = "")

See also create_data_link.pl, which is intended to work with the resulting
directory structure, and remove_data_links.sh
EOU

my $suffix="";
GetOptions('suffix=s' => \$suffix);

if (@ARGV < 2) {
  die $Usage;
}

my $ans = 1;

my $dir = pop(@ARGV);
system("mkdir -p $dir 2>/dev/null");

my @all_actual_storage = ();
foreach my $file (@ARGV) {
  push @all_actual_storage, File::Spec->rel2abs($file . "/" . $suffix);
}

my $index = 1;
foreach my $actual_storage (@all_actual_storage) {
  my $pseudo_storage = "$dir/$index";

  # If the symbolic link already exists, delete it.
  if (-l $pseudo_storage) {
    print STDERR "$0: link $pseudo_storage already exists, not overwriting.\n";
    $index++;
    next;
  }

  # Create the destination directory and make the link.
  system("mkdir -p $actual_storage 2>/dev/null");
  if ($? != 0) {
    print STDERR "$0: error creating directory $actual_storage\n";
    exit(1);
  }
  { # create a README file for easier deletion.
    open(R, ">$actual_storage/README.txt");
    my $storage_dir = File::Spec->rel2abs($dir);
    print R "# This directory is linked from $storage_dir, as part of Kaldi striped data\n";
    print R "# The full list of directories where this data resides is:\n";
    foreach my $d (@all_actual_storage) {
      print R "$d\n";
    }
    close(R);
  }
  my $ret = symlink($actual_storage, $pseudo_storage);

  # Process the returned values
  $ans = $ans && $ret;
  if (! $ret) {
    print STDERR "Error linking $actual_storage to $pseudo_storage\n";
  }

  $index++;
}

exit($ans == 1 ? 0 : 1);
