#!/usr/bin/perl -w

# In general, doing 
#  run.pl some.log a b c is like running the command a b c in
# the bash shell, and putting the standard error and output into some.log.
# To run parallel jobs (backgrounded on the host machine), you can do (e.g.)
#  run.pl JOB=1:4 some.JOB.log a b c JOB is like running the command a b c JOB
# and putting it in some.JOB.log, for each one. [Note: JOB can be any identifier].
# If any of the jobs fails, this script will fail.

# A typical example is:
#  run.pl some.log my-prog "--opt=foo bar" foo \|  other-prog baz
# and run.pl will run something like:
# ( my-prog '--opt=foo bar' foo |  other-prog baz ) >& some.log
# 
# Basically it takes the command-line arguments, quotes them
# as necessary to preserve spaces, and evaluates them with bash.
# In addition it puts the command line at the top of the log, and
# the start and end times of the command at the beginning and end.
# The reason why this is useful is so that we can create a different
# version of this program that uses a queueing system instead.

@ARGV < 2 && die "usage: run.pl log-file command-line arguments...";

$jobstart=1;
$jobend=1;
$qsub_opts=""; # These will be ignored.

# First parse an option like JOB=1:4, and any
# options that would normally be given to
# queue.pl, which we will just discard.

if (@ARGV > 0) {
  while (@ARGV >= 2 && $ARGV[0] =~ m:^-:) { # parse any options
    # that would normally go to qsub, but which will be ignored here.
    $switch = shift @ARGV;
    if ($switch eq "-V") {
      $qsub_opts .= "-V ";
    } else {
      $option = shift @ARGV;
      if ($switch eq "-sync" && $option =~ m/^[yY]/) {
        $qsub_opts .= "-sync "; # Note: in the
        # corresponding code in queue.pl it says instead, just "$sync = 1;".
      }
      $qsub_opts .= "$switch $option ";
      if ($switch eq "-pe") { # e.g. -pe smp 5
        $option2 = shift @ARGV;
        $qsub_opts .= "$option2 ";
      }
    }
  }
  if ($ARGV[0] =~ m/^([\w_][\w\d_]*)+=(\d+):(\d+)$/) { # e.g. JOB=1:10
    $jobname = $1;
    $jobstart = $2;
    $jobend = $3;
    shift;
    if ($jobstart > $jobend) {
      die "run.pl: invalid job range $ARGV[0]";
    }
  } elsif ($ARGV[0] =~ m/^([\w_][\w\d_]*)+=(\d+)$/) { # e.g. JOB=1.
    $jobname = $1;
    $jobstart = $2;
    $jobend = $2;
    shift;
  } elsif ($ARGV[0] =~ m/.+\=.*\:.*$/) {
    print STDERR "Warning: suspicious first argument to run.pl: $ARGV[0]\n";
  }
}

if ($qsub_opts ne "") {
  print STDERR "Warning: run.pl ignoring options \"$qsub_opts\"\n";
}

$logfile = shift @ARGV;

if (defined $jobname && $logfile !~ m/$jobname/ &&
    $jobend > $jobstart) {
  print STDERR "run.pl: you are trying to run a parallel job but "
    . "you are putting the output into just one log file ($logfile)\n";
  exit(1);
}

$cmd = "";

foreach $x (@ARGV) { 
    if ($x =~ m/^\S+$/) { $cmd .=  $x . " "; }
    elsif ($x =~ m:\":) { $cmd .= "'$x' "; }
    else { $cmd .= "\"$x\" "; } 
}


for ($jobid = $jobstart; $jobid <= $jobend; $jobid++) {
  $childpid = fork();
  if (!defined $childpid) { die "Error forking in run.pl (writing to $logfile)"; }
  if ($childpid == 0) { # We're in the child... this branch
    # executes the job and returns (possibly with an error status).
    if (defined $jobname) { 
      $cmd =~ s/$jobname/$jobid/g;
      $logfile =~ s/$jobname/$jobid/g;
    }
    system("mkdir -p `dirname $logfile` 2>/dev/null");
    open(F, ">$logfile") || die "Error opening log file $logfile";
    print F "# " . $cmd . "\n";
    print F "# Started at " . `date`;
    $starttime = `date +'%s'`;
    print F "#\n";
    close(F);

    # Pipe into bash.. make sure we're not using any other shell.
    open(B, "|bash") || die "Error opening shell command"; 
    print B "( " . $cmd . ") 2>>$logfile >> $logfile";
    close(B);                   # If there was an error, exit status is in $?
    $ret = $?;

    $endtime = `date +'%s'`;
    open(F, ">>$logfile") || die "Error opening log file $logfile (again)";
    $enddate = `date`;
    chop $enddate;
    print F "# Accounting: time=" . ($endtime - $starttime) . " threads=1\n";
    print F "# Ended (code $ret) at " . $enddate . ", elapsed time " . ($endtime-$starttime) . " seconds\n";
    close(F);
    exit($ret == 0 ? 0 : 1);
  }
}

$ret = 0;
$numfail = 0;
for ($jobid = $jobstart; $jobid <= $jobend; $jobid++) {
  $r = wait();
  if ($r == -1) { die "Error waiting for child process"; } # should never happen.
  if ($? != 0) { $numfail++; $ret = 1; } # The child process failed.
}

if ($ret != 0) {
  $njobs = $jobend - $jobstart + 1;
  if ($njobs == 1) { 
    if (defined $jobname) {
      $logfile =~ s/$jobname/$jobstart/; # only one numbered job, so replace name with
                                         # that job.
    }
    print STDERR "run.pl: job failed, log is in $logfile\n";
    if ($logfile =~ m/JOB/) {
      print STDERR "run.pl: probably you forgot to put JOB=1:\$nj in your script.";
    }
  }
  else {
    $logfile =~ s/$jobname/*/g;
    print STDERR "run.pl: $numfail / $njobs failed, log is in $logfile\n";
  }
}


exit ($ret);
