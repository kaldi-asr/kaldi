#!/usr/bin/perl

# 
# If the command-line argument is 4, this script prints "0 1 2 3";
# If the command-line argument is 20, it prints 10 through 29.
# These numbers are used to allocate the names of split-up data 
# directories.  We make the names this way so that when we use globs
# to represent filenames with these numbers in, they remain in the correct
# order.  We don't use leading zeros for this purpose because they don't
# interact well with bash variable name indexing.

if (@ARGV != 2 || $ARGV[0] !~ m:^\d+$: || $ARGV[0] < 1) {
    die "Invalid command-line arguments (expect the number of splits)";
}

$n = $ARGV[0];
$start = 0;

if ($n > 10) { $start = 10; }
if ($n > 90) { $start = 100; }
if ($n > 900) { $start = 1000; }
if ($n > 9000) { $start = 10000; }

for ($x = $start; $x < $start + $n; $x++) {
    print $x . " ";
}
print "\n";

