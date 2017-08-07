#!/usr/bin/perl

# Copyright  2017  Johns Hopkins University (author: Daniel Povey)
# License: Apache 2.0.

# When run with usage like:
# cat foo | rnnlm/distribute_lines.pl 1.txt 2.txt 3.txt
# this program will distribute the lines from the input round-robin and
# *append* them to the files 1.txt, 2.txt and 3.txt.



$prefix_str == "";

if ((scalar @ARGV) > 1 && $ARGV[0] eq "--prefix") {
  shift @ARGV;
  $prefix_str = $ARGV[0];
  shift @ARGV;
}

if ((scalar @ARGV) == 0) {
  die "Usage: cat data | split_data.pl [--prefix prefix_str] <file1> <file2> .. <fileN>\n" .
     "e.g. cat foo | split_data.pl 1.txt 2.txt 3.txt\n" .
     "This program distributes the input lines round-robin to the specified files,\n" .
     "after prepending the prefix 'prefix_str', if specified, to each line.";
}



@outputs = ();  # contains filehandles.

# open filehandles for writing.
for ($n = 0; $n < scalar @ARGV; $n++) {
  my $fh = "O$n";
  # Open for appending.
  if (!open($fh, ">>", "$ARGV[$n]")) {
    die "$0: failed to open $n'th filehandle, with name $ARGV[$n].  If it's a max-open-filehandles issue, " .
        "we may have to refactor this script to write first to temporary files if " .
        "there are too many outputs.  As a workaround, aim for larger <target-words-per-split> " .
        "in get_num_splits.sh.";
  }
  push @outputs, $fh;
}

$num_filehandles = scalar @outputs;
$i = 0;
while (<STDIN>) {
  my $line = $_;
  my $fh = $outputs[$i % $num_filehandles];
  if (! print $fh "$prefix_str$line") {
    die "$0: write error.  Disk full?";
  }
  $i++;
}

foreach $fh (@outputs) {
  if (! close $fh) {
    die "$0: error closing filehandle.  Disk full?";
  }
}
