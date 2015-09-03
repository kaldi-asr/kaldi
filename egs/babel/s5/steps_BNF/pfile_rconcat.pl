#!/usr/bin/env perl

# Copyright 2013  Karlsruhe Institute of Technology (Author: Jonas Gehring)
# Apache 2.0.
#
# Concatenates random subsets of a list of pfiles
#

use Getopt::Long;
use List::Util qw(shuffle);
use File::Basename;

# Prints usage information
sub usage() {
	print STDERR << "EOF";
Usage: pfile_rconcat <options> <input-files>
Where <options> include:
-t                       tmpdir
-h                       print this message
-o <output-file>,<ratio> output pfile with corresponding ratio
EOF
	exit;
}

# Main program
my @outfiles, $help, $tmpdir;
GetOptions("o=s" => \@outfiles, "t=s" => \$tmpdir, "h!" => \$help);
usage() if $help;
if (@ARGV == 0) {
	die("No input files specified!");
}
my @infiles = shuffle(@ARGV);
if ($tmpdir ne "") { $tmpdir = $tmpdir . "/"; }


# Produce output files
my $remain = 1.0;
my $total = @infiles;
foreach (@outfiles) {
	my @parts = split(/,/);
	my $outfile = @parts[0];
	my $outbase = basename($outfile, ".pfile");
	my $ratio = @parts[1];
	my $n = int($total * $ratio);

	if (not $ratio) {
		$ratio = $remain;
		$n = @infiles;
	}

	$remain = $remain - $ratio;
	if ($remain < 0.0) {
		die("Ratio exceeds total amount: $ratio");
	}

	my $i = 0, $fno = 0;
	my @list = ();
	my $cmd_line;
	while ($n > 0) {
		push(@list, shift(@infiles));
		$n--;
		if (@list == 1024) {
			$cmd_line = sprintf("pfile_concat -o ${tmpdir}%s_$fno.pfile @list", $outbase);
			system($cmd_line) == 0 or die("Error running command $cmd_line");
			@list = ();
			$fno++;
		}
	}

	# Concatenate partial files?
	my $cleanup = undef;
	if ($fno > 0) {
		$cmd_line = sprintf("pfile_concat -o ${tmpdir}%s_$fno.pfile @list", $outbase);
		system($cmd_line) == 0 or die("Error running command $cmd_line");
		@list = ();
		while ($fno >= 0) {
			push(@list, sprintf("${tmpdir}%s_$fno.pfile", $outbase));
			$fno--;
		}
		$cleanup = 1;
	}

	$cmd_line = sprintf("pfile_concat -o %s @list", $outfile);
	system($cmd_line) == 0 or die("Error running command $cmd_line");

	if ($cleanup) {
		foreach $file (@files) {
			unlink($file);
		}
	}
}
