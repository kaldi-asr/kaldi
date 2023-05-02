#!/usr/bin/env perl
use strict;
use warnings;

# Copyright 2018  Johns Hopkins University (Author: Daniel Povey).
# Apache 2.0.

use File::Basename;
use Cwd;
use Getopt::Long;


# retry.pl is a wrapper for queue.pl.  It can be used to retry jobs that failed,
# e.g. if your command line was "queue.pl [args]", you can replace that
# with "retry.pl queue.pl [args]" and it will retry jobs that failed.


my $num_tries = 2;

sub print_usage() {
  print STDERR
    "Usage: retry.pl  <some-other-wrapper-script> <rest-of-command>\n" .
    "  e.g.:  retry.pl [options] queue.pl foo.log do_something\n" .
    "This will retry jobs that failed (only once)\n" .
    "Options:\n" .
    "      --num-tries <n>        # default: 2\n";
  exit 1;
}

if ($ARGV[0] eq "--num-tries") {
  shift;
  $num_tries =  $ARGV[0] + 0;
  if ($num_tries < 1) {
    die "$0: invalid option --num-tries $ARGV[0]";
  }
  shift;
}

if (@ARGV < 3) {
  print_usage();
}


sub get_log_file {
  my $n;
  # First just look for the first command-line arg that ends in ".log".  If that
  # exists, it's almost certainly the log file.
  for ($n = 1; $n < @ARGV; $n++) {
    if ($ARGV[$n] =~ m/\.log$/) {
      return $ARGV[$n];
    }
  }
  for ($n = 1; $n < @ARGV; $n++) {
    # If this arg isn't of the form "-some-option', and isn't of the form
    # "JOB=1:10", and the previous arg wasn't of the form "-some-option", and this
    # isn't just a number (note: the 'not-a-number' things is mostly to exclude
    # things like the 5 in "-pe smp 5" which is an older but still-supported
    # option to queue.pl)... then assume it's a log file.
    if ($ARGV[$n] !~ m/^-=/ &&  $ARGV[$n] !~ m/=/ && $ARGV[$n] !~ m/^\d+$/ &&
        $ARGV[$n-1] !~ m/^-/) {
      return $ARGV[$n];
    }
  }
  print STDERR "$0: failed to parse log-file name from args:" . join(" ", @ARGV);
  exit(1);
}


my $log_file = get_log_file();
my $return_status;

for (my $n = 1; $n <= $num_tries; $n++) {
  system(@ARGV);
  $return_status = $?;
  if ($return_status == 0) {
    exit(0);  # The command succeeded.  We return success.
  } elsif ($return_status != 256) {
    # The command did not "die normally".  When queue.pl and similar scripts
    # detect a normal error, they exit(1), which becomes a status of 256
    # in perl's $? variable.
    # See http://perldoc.perl.org/perlvar.html#%24CHILD_ERROR for more info.
    # An example of an abnormal death that would cause us to want to exit
    # immediately, is when the user does ctrl-c or KILLs the script,
    # which gets caught by 'caught_signal' in queue.pl and causes that program
    # to return with exit status 2.
    exit(1);
  }


  if ($n < $num_tries) {
    if (! -f $log_file) {
      # $log_file doesn't exist as a file.  Maybe it was an array job.
      # This script doesn't yet support array jobs.  We just give up.
      # Later on we might want to figure out which array jobs failed
      # and have to be rerun, but for now we just die.
      print STDERR "$0: job failed and log file $log_file does not exist (array job?).\n";
    } else {
      rename($log_file, $log_file . ".bak");
      print STDERR "$0: job failed; renaming log file to ${log_file}.bak and rerunning\n";
    }
  }
}

print STDERR "$0: job failed $num_tries times; log is in $log_file\n";
exit(1);
