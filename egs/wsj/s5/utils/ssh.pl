#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter

use Cwd;
use File::Basename;

# This program is like run.pl except rather than just running on a local
# machine, it can be configured to run on remote machines via ssh.
# It requires that you have set up passwordless access to those machines,
# and that Kaldi is running from a location that is accessible via the
# same path on those machines (presumably via an NFS mount).
#
# It looks for a file .queue/machines that should have, on each line, the name
# of a machine that you can ssh to (which may include this machine).  It doesn't
# have to be a fully qualified name.
#
# Later we may extend this so that on each line of .queue/machines you
# can specify various resources that each machine has, such as how
# many slots and how much memory, and make it wait if machines are 
# busy.  But for now it simply ssh's to a machine from those in the list.

# The command-line interface of this program is the same as run.pl;
# see run.pl for more information about the usage.


@ARGV < 2 && die "usage: ssh.pl log-file command-line arguments...";

$jobstart = 1;
$jobend = 1;
$qsub_opts=""; # These will be ignored.

# First parse an option like JOB=1:4, and any
# options that would normally be given to 
# ssh.pl, which we will just discard.

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
    if ($jobstart <= 0) {
      die "run.pl: invalid job range $ARGV[0], start must be strictly positive (this is required for GridEngine compatibility)";
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
  print STDERR "Warning: ssh.pl ignoring options \"$qsub_opts\"\n";
}

{ # Read .queue/machines
  if (!open(Q, "<.queue/machines")) {
    print STDERR "ssh.pl: expected the file .queue/machines to exist.\n";
    exit(1);
  }
  @machines = ();
  while (<Q>) {
    chop;
    if ($_ ne "") {
      @A = split;
      if (@A != 1) {
        die "ssh.pl: bad line '$_' in .queue/machines.";
      }
      if ($A[0] !~ m/^[a-z0-9\.\-]+/) {
        die "ssh.pl: invalid machine name '$A[0]'";
      }
      push @machines, $A[0];
    }
  }
  if (@machines == 0) {   die "ssh.pl: no machines listed in .queue/machines";  }
}

$logfile = shift @ARGV;

if (defined $jobname && $logfile !~ m/$jobname/ &&
    $jobend > $jobstart) {
  print STDERR "ssh.pl: you are trying to run a parallel job but "
    . "you are putting the output into just one log file ($logfile)\n";
  exit(1);
}

{
  $offset = 0;  # $offset will be an offset added to any index from the job-id
                # specified if the user does JOB=1:10.  The main point of this is
                # that there are instances where a script will manually submit a
                # number of jobs to the queue, e.g. with log files foo.1.log,
                # foo.2.log and so on, and we don't want all of these to go
                # to the first machine.
  @A = split(".", basename($logfile));
  # if $logfile looks like foo.9.log, add 9 to $offset.
  foreach $a (@A) {  if ($a =~ m/^\d+$/) { $offset += $a; } }
}

$cmd = "";

foreach $x (@ARGV) { 
    if ($x =~ m/^\S+$/) { $cmd .=  $x . " "; }
    elsif ($x =~ m:\":) { $cmd .= "'$x' "; }
    else { $cmd .= "\"$x\" "; } 
}


for ($jobid = $jobstart; $jobid <= $jobend; $jobid++) {
  $childpid = fork();
  if (!defined $childpid) { die "Error forking in ssh.pl (writing to $logfile)"; }
  if ($childpid == 0) {
    # We're in the child... this branch executes the job and returns (possibly
    # with an error status).
    if (defined $jobname) {
      $cmd =~ s/$jobname/$jobid/g;
      $logfile =~ s/$jobname/$jobid/g;
    }
    { # work out the machine to ssh to.
      $local_offset = $offset + $jobid - 1;  # subtract 1 since jobs never start
                                             # from 0; we'd like the first job
                                             # to normally run on the first
                                             # machine.
      $num_machines = scalar @machines;
      # in the next line, the "+ $num_machines" is in case $local_offset is
      # negative, to ensure the modulus is calculated in the mathematical way, not
      # in the C way where (negative number % positive number) is negative.
      $machines_index = ($local_offset + $num_machines) % $num_machines;
      $machine = $machines[$machines_index];
    }
    if (!open(S, "|ssh $machine bash")) {
      print STDERR "ssh.pl failed to ssh to $machine";
      exit(1);  # exits from the forked process within ssh.pl.
    }
    $cwd = getcwd();
    $logdir = dirname($logfile);
    # Below, we're printing into ssh which has opened a bash session; these are
    # bash commands.
    print S "set -e\n";  # if any of the later commands fails, we want it to exit.
    print S "cd $cwd\n";
    print S ". ./path.sh\n";
    print S "mkdir -p $logdir\n";
    print S "time1=\`date +\"%s\"\`\n";
    print S "( echo '#' Running on \`hostname\`\n";
    print S "  echo '#' Started at \`date\`\n";
    print S "  echo -n '# '; cat <<EOF\n";
    print S "$cmd\n";
    print S "EOF\n";
    print S ") >$logfile\n";
    print S "set +e\n";  # we don't want bash to exit if the next line fails.
    # in the next line, || true means allow this one to fail and not have bash exit immediately.
    print S " ( $cmd ) 2>>$logfile >>$logfile\n"; 
    print S "ret=\$?\n";
    print S "set -e\n"; # back into mode where it will exit on error.
    print S "time2=\`date +\"%s\"\`\n";
    print S "echo '#' Accounting: time=\$((\$time2-\$time1)) threads=1 >>$logfile\n";
    print S "echo '#' Finished at \`date\` with status \$ret >>$logfile\n";
    print S "exit \$ret";  # return with the status the command exited with.
    $ret = close(S);
    $ssh_return_status = $?;
    # see http://perldoc.perl.org/functions/close.html for explanation of return
    # status of close() and the variables it sets.
    if (! $ret && $! != 0) { die "ssh.pl: unexpected problem ssh'ing to machine $machine"; }
    if ($ssh_return_status != 0) { exit(1); } # exit with error status from this forked process.
    else { exit(0); } # else exit with non-error status.
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
    print STDERR "ssh.pl: job failed, log is in $logfile\n";
    if ($logfile =~ m/JOB/) {
      print STDERR "run.pl: probably you forgot to put JOB=1:\$nj in your script.";
    }
  }
  else {
    $logfile =~ s/$jobname/*/g;
    print STDERR "ssh.pl: $numfail / $njobs failed, log is in $logfile\n";
  }
}


exit ($ret);
