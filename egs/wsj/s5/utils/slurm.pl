#!/usr/bin/perl -w

# In general, doing 
#  slurm.pl some.log a b c 
# is like running the command a b c as an interactive SLURM job, and putting the 
# standard error and output into some.log.
# It is a de-facto-mimicry of run.pl, with the difference, that it allocates the
# jobs on a slurm cluster, using SLURM's salloc.
#
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
#
# You can also specify command-line options that are passed to SLURM's
# salloc, e.g.:
#
# slurm.pl -p long -c 4 --mem=16g JOB=1:4 some.JOB.log a b c JOB
#
# The options "-p long -c 4 --mem=16g" are passed to salloc. In general, all
# options of form "-foo something" and "--fii" (and also "-V") are recognized
# as options that must be passed to salloc (yes, this is a bit hacky).
#
# In addition to that, the script also converts an option like '-pe smp N'
# to a form '-c N'. This is because some Kaldi's scripts use this SGE syntax
# to specify the number of CPU cores for a job.
#

@ARGV < 2 && die "usage: slurm.pl log-file command-line arguments...";

$jobstart=1;
$jobend=1;
$queue_opts = "";

# First parse an option like JOB=1:4

for ($x = 1; $x <= 3; $x++) { # This for-loop is to 
  # allow the JOB=1:n option to be interleaved with the
  # options to qsub.
  while (@ARGV >= 2 && $ARGV[0] =~ m:^-:) {
    $switch = shift @ARGV;
    if ($switch =~ m:--:) {  # e.g. --gres=gpu:1
      $queue_opts .= " $switch ";
    } elsif ($switch eq "-V") {
      $queue_opts .= "-V ";
    } else {
      $option = shift @ARGV;
      # Override '-pe smp 5' like option for SGE
      # for SLURM, the equivalent is '-c 5'.
      # This option is hard-coded in some training scripts
      if ($switch eq "-pe" && $option eq "smp") { # e.g. -pe smp 5
        $option2 = shift @ARGV;
        $nof_threads = $option2;
        $queue_opts .= "-c $nof_threads ";
        print STDERR "slurm.pl: converted SGE option '-pe smp $nof_threads' to SLURM option '-c $nof_threads'\n"
      } else {  
        $queue_opts .= "$switch $option ";
      }
    }

  }
  if ($ARGV[0] =~ m/^([\w_][\w\d_]*)+=(\d+):(\d+)$/) {
    $jobname = $1;
    $jobstart = $2;
    $jobend = $3;
    shift;
    if ($jobstart > $jobend) {
      die "queue.pl: invalid job range $ARGV[0]";
    }
    if ($jobstart <= 0) {
      die "run.pl: invalid job range $ARGV[0], start must be strictly positive (this is required for GridEngine compatibility).";
    }
  } elsif ($ARGV[0] =~ m/^([\w_][\w\d_]*)+=(\d+)$/) { # e.g. JOB=1.
    $jobname = $1;
    $jobstart = $2;
    $jobend = $2;
    shift;
  } elsif ($ARGV[0] =~ m/.+\=.*\:.*$/) {
    print STDERR "Warning: suspicious first argument to queue.pl: $ARGV[0]\n";
  }
}

$logfile = shift @ARGV;

if (defined $jobname && $logfile !~ m/$jobname/ &&
    $jobend > $jobstart) {
  print STDERR "slurm.pl: you are trying to run a parallel job but "
    . "you are putting the output into just one log file ($logfile)\n";
  exit(1);
}

#
# Work out the command; quote escaping is done here.
# Note: the rules for escaping stuff are worked out pretty
# arbitrarily, based on what we want it to do.  Some things that
# we pass as arguments to queue.pl, such as "|", we want to be
# interpreted by bash, so we don't escape them.  Other things,
# such as archive specifiers like 'ark:gunzip -c foo.gz|', we want
# to be passed, in quotes, to the Kaldi program.  Our heuristic
# is that stuff with spaces in should be quoted.  This doesn't
# always work.
#
$cmd = "";

foreach $x (@ARGV) { 
  if ($x =~ m/^\S+$/) { $cmd .= $x . " "; } # If string contains no spaces, take
                                            # as-is.
  elsif ($x =~ m:\":) { $cmd .= "'\''$x'\'' "; } # else if no dbl-quotes, use single
  else { $cmd .= "\"$x\" "; }  # else use double.
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
    $startdate=`date`;
    chop $startdate;
    print F "# Invoked at " . $startdate . " from " . `hostname`;
    $starttime = `date +'%s'`;
    print F "#\n";
    close(F);

    # Pipe into bash.. make sure we're not using any other shell.

    open(B, "|-", "salloc $queue_opts srun bash") || die "Error opening shell command";    
    print B "echo '#' Started at `date` on `hostname` 2>>$logfile >> $logfile;";
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
