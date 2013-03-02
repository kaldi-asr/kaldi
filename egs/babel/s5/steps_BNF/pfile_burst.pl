#!/usr/bin/perl

# Copyright 2013  Karlsruhe Institute of Technology (Author: Jonas Gehring)
# Apache 2.0.
#
# Bursts a pfile into equally-sized parts
#

use Getopt::Std;
use List::Util qw/min/;

# Prints usage information
sub usage() {
	print STDERR << "EOF";
Usage: pfile_burst <options>
Where <options> include:
-h               print this message
-i <file-name>   input pfile
-o <folder-name> output folder 
-s <num>         number of sentences per split
EOF
	exit;
}

# Main program
my $opt_string = "hi:o:s:";
my %opt;
getopts("$opt_string", \%opt) or usage();
usage() if $opt{h};

# Read number of sentences of input file
my $num_sents = 0;
open(IN, "pfile_info $opt{i} |") or die("Unable to open input file");
while (<IN>) {
	if (m/([0-9]+) sentences/) {
		$num_sents = $1;
	}
}
if ($num_sents < 1) {
	die("Number of sentences is $num_sents; empty pfile?");
}

# Select sentence ranges
my $index = 0;
my $cmd_line;
while ($index < $num_sents) {
	$cmd_line = sprintf("pfile_select -i $opt{i} -o $opt{o}/%06d.pfile -sr $index:%d", $index / $opt{s}, min($index + $opt{s}, $num_sents) - 1);
	system($cmd_line) == 0 or die("Error running command");
	$index = $index + $opt{s};
}
