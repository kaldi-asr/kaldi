#!/usr/bin/perl -w

# In general, doing
#  slurm.pl some.log a b c
# is like running the command a b c as an interactive SLURM job, and putting the
# standard error and output into some.log.
# It is a de-facto-mimicry of run.pl, with the difference, that it allocates the
# jobs on a slurm cluster.  The calling script (e.g. decode.sh) should have the
# required allocation, e.g.
#   $ sbatch -n 40 steps/decode.sh --nj 40 --cmd utils/slurm.pl ...
# The benefit compared to qsub.pl is that there is no active wait involved, as
# this script waits on all forked processes to finish.
# To run parallel jobs (backgrounded on the host machine), you can do (e.g.)
#  slurm.pl JOB=1:4 some.JOB.log a b c JOB is like running the command a b c JOB
# and putting it in some.JOB.log, for each one. [Note: JOB can be any identifier].
# If any of the jobs fails, this script will fail.

# A typical example is:
#  slurm.pl some.log my-prog "--opt=foo bar" foo \|  other-prog baz
# and slurm.pl will run something like:
# ( my-prog '--opt=foo bar' foo |  other-prog baz ) >& some.log
# 
# Basically it takes the command-line arguments, quotes them
# as necessary to preserve spaces, and evaluates them with bash.
# In addition it puts the command line at the top of the log, and
# the start and end times of the command at the beginning and end.
# The reason why this is useful is so that we can create a different
# version of this program that uses a queueing system instead.

@ARGV < 2 && die "usage: slurm.pl log-file command-line arguments...";

$jobstart=1;
$jobend=1;

# First parse an option like JOB=1:4

if (@ARGV > 0) {
  if ($ARGV[0] =~ m/^([\w_][\w\d_]*)+=(\d+):(\d+)$/) {
    $jobname = $1;
    $jobstart = $2;
    $jobend = $3;
    shift;
    if ($jobstart > $jobend) {
      die "slurm.pl: invalid job range $ARGV[0]";
    }
  } elsif ($ARGV[0] =~ m/^([\w_][\w\d_]*)+=(\d+)$/) { # e.g. JOB=1.
    $jobname = $1;
    $jobstart = $2;
    $jobend = $2;
    shift;
  } elsif ($ARGV[0] =~ m/.+\=.*\:.*$/) {
    print STDERR "Warning: suspicious first argument to slurm.pl: $ARGV[0]\n";
  }
}

$logfile = shift @ARGV;

if (defined $jobname && $logfile !~ m/$jobname/ &&
    $jobend > $jobstart) {
  print STDERR "slurm.pl: you are trying to run a parallel job but "
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
  if (!defined $childpid) { die "Error forking in slurm.pl (writing to $logfile)"; }
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

    open(B, "|-", "srun -N 1 -n 1 bash") || die "Error opening shell command"; 
    print B "( " . $cmd . ") 2>>$logfile >> $logfile";
    close(B);                   # If there was an error, exit status is in $?
    $ret = $?;

    $endtime = `date +'%s'`;
    open(F, ">>$logfile") || die "Error opening log file $logfile (again)";
    $enddate = `date`;
    chop $enddate;
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
    print STDERR "slurm.pl: job failed, log is in $logfile\n";
    if ($logfile =~ m/JOB/) {
      print STDERR "slurm.pl: probably you forgot to put JOB=1:\$nj in your script.\n";
    }
  }
  else {
    $logfile =~ s/$jobname/*/g;
    print STDERR "slurm.pl: $numfail / $njobs failed, log is in $logfile\n";
  }
}


exit ($ret);
