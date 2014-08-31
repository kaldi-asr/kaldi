#!/usr/bin/perl
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).
# Apache 2.0.
use File::Basename;
use Cwd;

# queue.pl has the same functionality as run.pl, except that
# it runs the job in question on the queue (Sun GridEngine).
# This version of queue.pl uses the task array functionality
# of the grid engine.  Note: it's different from the queue.pl
# in the s4 and earlier scripts.

$qsub_opts = "";
$sync = 0;
$nof_threads=1;

for ($x = 1; $x <= 3; $x++) { # This for-loop is to 
  # allow the JOB=1:n option to be interleaved with the
  # options to qsub.
  while (@ARGV >= 2 && $ARGV[0] =~ m:^-:) {
    $switch = shift @ARGV;
    if ($switch eq "-V") {
      $qsub_opts .= "-V ";
    } else {
      $option = shift @ARGV;
      if ($switch eq "-sync" && $option =~ m/^[yY]/) {
        $sync = 1;
      }
      $qsub_opts .= "$switch $option ";
      if ($switch eq "-pe") { # e.g. -pe smp 5
        $option2 = shift @ARGV;
        $qsub_opts .= "$option2 ";
        $nof_threads = $option2;
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
  } elsif ($ARGV[0] =~ m/^([\w_][\w\d_]*)+=(\d+)$/) { # e.g. JOB=1.
    $jobname = $1;
    $jobstart = $2;
    $jobend = $2;
    shift;
  } elsif ($ARGV[0] =~ m/.+\=.*\:.*$/) {
    print STDERR "Warning: suspicious first argument to queue.pl: $ARGV[0]\n";
  }
}


if (@ARGV < 2) {
  print STDERR
   "Usage: queue.pl [options to qsub] [JOB=1:n] log-file command-line arguments...\n" .
   "e.g.: queue.pl foo.log echo baz\n" .
   " (which will echo \"baz\", with stdout and stderr directed to foo.log)\n" .
   "or: queue.pl -q all.q\@xyz foo.log echo bar \| sed s/bar/baz/ \n" .
   " (which is an example of using a pipe; you can provide other escaped bash constructs)\n" .
   "or: queue.pl -q all.q\@qyz JOB=1:10 foo.JOB.log echo JOB \n" .
   " (which illustrates the mechanism to submit parallel jobs; note, you can use \n" .
   "  another string other than JOB)\n" .
   "Note: if you pass the \"-sync y\" option to qsub, this script will take note\n" .
   "and change its behavior.  Otherwise it uses qstat to work out when the job finished\n";
  exit 1;
}

$cwd = getcwd();
$logfile = shift @ARGV;

if (defined $jobname && $logfile !~ m/$jobname/
    && $jobend > $jobstart) {
  print STDERR "queue.pl: you are trying to run a parallel job but "
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
  elsif ($x =~ m:\":) { $cmd .= "'$x' "; } # else if no dbl-quotes, use single
  else { $cmd .= "\"$x\" "; }  # else use double.
}

#
# Work out the location of the script file, and open it for writing.
#
$dir = dirname($logfile);
$base = basename($logfile);
$qdir = "$dir/q";
$qdir =~ s:/(log|LOG)/*q:/q:; # If qdir ends in .../log/q, make it just .../q.
$queue_logfile = "$qdir/$base";

if (!-d $dir) { system "mkdir -p $dir 2>/dev/null"; } # another job may be doing this...
if (!-d $dir) { die "Cannot make the directory $dir\n"; }
# make a directory called "q",
# where we will put the log created by qsub... normally this doesn't contain
# anything interesting, evertyhing goes to $logfile.
if (! -d "$qdir") { 
  system "mkdir $qdir 2>/dev/null";
  sleep(5); ## This is to fix an issue we encountered in denominator lattice creation,
  ## where if e.g. the exp/tri2b_denlats/log/15/q directory had just been
  ## created and the job immediately ran, it would die with an error because nfs
  ## had not yet synced.  I'm also decreasing the acdirmin and acdirmax in our
  ## NFS settings to something like 5 seconds.
} 

if (defined $jobname) { # It's an array job.
  $queue_array_opt = "-t $jobstart:$jobend"; 
  $logfile =~ s/$jobname/\$SGE_TASK_ID/g; # This variable will get 
  # replaced by qsub, in each job, with the job-id.
  $cmd =~ s/$jobname/\$SGE_TASK_ID/g; # same for the command...
  $queue_logfile =~ s/\.?$jobname//; # the log file in the q/ subdirectory
  # is for the queue to put its log, and this doesn't need the task array subscript
  # so we remove it.
}

# queue_scriptfile is as $queue_logfile [e.g. dir/q/foo.log] but
# with the suffix .sh.
$queue_scriptfile = $queue_logfile;
($queue_scriptfile =~ s/\.[a-zA-Z]{1,5}$/.sh/) || ($queue_scriptfile .= ".sh");
if ($queue_scriptfile !~ m:^/:) {
  $queue_scriptfile = $cwd . "/" . $queue_scriptfile; # just in case.
}

# We'll write to the standard input of "qsub" (the file-handle Q),
# the job that we want it to execute.
# Also keep our current PATH around, just in case there was something
# in it that we need (although we also source ./path.sh)

$syncfile = "$qdir/done.$$";

system("rm $queue_logfile $syncfile 2>/dev/null");
#
# Write to the script file, and then close it.
#
open(Q, ">$queue_scriptfile") || die "Failed to write to $queue_scriptfile";

print Q "#!/bin/bash\n";
print Q "cd $cwd\n";
print Q ". ./path.sh\n";
print Q "( echo '#' Running on \`hostname\`\n";
print Q "  echo '#' Started at \`date\`\n";
print Q "  echo -n '# '; cat <<EOF\n";
print Q "$cmd\n"; # this is a way of echoing the command into a comment in the log file,
print Q "EOF\n"; # without having to escape things like "|" and quote characters.
print Q ") >$logfile\n";
print Q "time1=\`date +\"%s\"\`\n";
print Q " ( $cmd ) 2>>$logfile >>$logfile\n";
print Q "ret=\$?\n";
print Q "time2=\`date +\"%s\"\`\n";
print Q "echo '#' Accounting: time=\$((\$time2-\$time1)) threads=$nof_threads >>$logfile\n";
print Q "echo '#' Finished at \`date\` with status \$ret >>$logfile\n";
print Q "[ \$ret -eq 137 ] && exit 100;\n"; # If process was killed (e.g. oom) it will exit with status 137;
  # let the script return with status 100 which will put it to E state; more easily rerunnable.
if (!defined $jobname) { # not an array job
  print Q "touch $syncfile\n"; # so we know it's done.
} else {
  print Q "touch $syncfile.\$SGE_TASK_ID\n"; # touch a bunch of sync-files.
}
print Q "exit \$[\$ret ? 1 : 0]\n"; # avoid status 100 which grid-engine
print Q "## submitted with:\n";       # treats specially.
$qsub_cmd = "qsub -S /bin/bash -v PATH -cwd -j y -o $queue_logfile $qsub_opts $queue_array_opt $queue_scriptfile >>$queue_logfile 2>&1";
print Q "# $qsub_cmd\n";
if (!close(Q)) { # close was not successful... || die "Could not close script file $shfile";
  die "Failed to close the script file (full disk?)";
}

$ret = system ($qsub_cmd);
if ($ret != 0) {
  if ($sync && $ret == 256) { # this is the exit status when a job failed (bad exit status)
    if (defined $jobname) { $logfile =~ s/\$SGE_TASK_ID/*/g; }
    print STDERR "queue.pl: job writing to $logfile failed\n";
  } else {
    print STDERR "queue.pl: error submitting jobs to queue (return status was $ret)\n";
    print STDERR "queue log file is $queue_logfile, command was $qsub_cmd\n";
    print STDERR `tail $queue_logfile`;
  }
  exit(1);
}

if (! $sync) { # We're not submitting with -sync y, so we
  # need to wait for the jobs to finish.  We wait for the
  # sync-files we "touched" in the script to exist.
  @syncfiles = ();
  if (!defined $jobname) { # not an array job.
    push @syncfiles, $syncfile;
  } else {
    for ($jobid = $jobstart; $jobid <= $jobend; $jobid++) {
      push @syncfiles, "$syncfile.$jobid";
    }
  }
  # We will need the sge_job_id, to check that job still exists
  { # Get the SGE job-id from the log file in q/
    open(L, "<$queue_logfile") || die "Error opening log file $queue_logfile";
    undef $sge_job_id;
    while (<L>) {
      if (m/Your job\S* (\d+)[. ].+ has been submitted/) {
        if (defined $sge_job_id) {
          die "Error: your job was submitted more than once (see $queue_logfile)";
        } else {
          $sge_job_id = $1;
        }
      }
    }
    close(L);
    if (!defined $sge_job_id) {
      die "Error: log file $queue_logfile does not specify the SGE job-id.";
    }
  }
  $check_sge_job_ctr=1;
  #
  $wait = 0.1;
  foreach $f (@syncfiles) {
    # wait for them to finish one by one.
    while (! -f $f) {
      sleep($wait);
      $wait *= 1.2;
      if ($wait > 3.0) {
        $wait = 3.0; # never wait more than 3 seconds.
        if (rand() > 0.5) {
          system("touch $qdir/.kick");
        } else {
          system("rm $qdir/.kick 2>/dev/null");
        }
        # This seems to kick NFS in the teeth to cause it to refresh the
        # directory.  I've seen cases where it would indefinitely fail to get
        # updated, even though the file exists on the server.
        system("ls $qdir >/dev/null");
      }

      # Check that the job exists in SGE. Job can be killed if duration 
      # exceeds some hard limit, or in case of a machine shutdown. 
      if (($check_sge_job_ctr++ % 10) == 0) { # Don't run qstat too often, avoid stress on SGE.
        if ( -f $f ) { next; }; #syncfile appeared: OK.
        $ret = system("qstat -j $sge_job_id >/dev/null 2>/dev/null");
        # system(...) : To get the actual exit value, shift $ret right by eight bits.
        if ($ret>>8 == 1) {     # Job does not seem to exist
          # Don't consider immediately missing job as error, first wait some  
          # time to make sure it is not just delayed creation of the syncfile.

          sleep(3);
          # Sometimes NFS gets confused and thinks it's transmitted the directory
          # but it hasn't, due to timestamp issues.  Changing something in the
          # directory will usually fix that.
          system("touch $qdir/.kick");
          system("rm $qdir/.kick 2>/dev/null");
          if ( -f $f ) { next; }   #syncfile appeared, ok
          sleep(7);
          system("touch $qdir/.kick");
          sleep(1);
          system("rm $qdir/.kick 2>/dev/null");
          if ( -f $f ) {  next; }   #syncfile appeared, ok
          sleep(60);
          system("touch $qdir/.kick");
          sleep(1);
          system("rm $qdir/.kick 2>/dev/null");
          if ( -f $f ) { next; }  #syncfile appeared, ok
          $f =~ m/\.(\d+)$/ || die "Bad sync-file name $f";
          $job_id = $1;
          if (defined $jobname) {
            $logfile =~ s/\$SGE_TASK_ID/$job_id/g;
          }
          $last_line = `tail -n 1 $logfile`;
          if ($last_line =~ m/status 0$/ && (-M $logfile) < 0) {
            # if the last line of $logfile ended with "status 0" and
            # $logfile is newer than this program [(-M $logfile) gives the
            # time elapsed between file modification and the start of this
            # program], then we assume the program really finished OK,
            # and maybe something is up with the file system.
            print STDERR "**queue.pl: syncfile $f was not created but job seems\n" .
              "**to have finished OK.  Probably your file-system has problems.\n" .
              "**This is just a warning.\n";
            last;
          } else {
            chop $last_line;
            print STDERR "queue.pl: Error, unfinished job no " .
              "longer exists, log is in $logfile, last line is '$last_line'" .
              "syncfile is $f, return status of qstat was $ret\n" .
              "Possible reasons: a) Exceeded time limit? -> Use more jobs!" .
              " b) Shutdown/Frozen machine? -> Run again!\n";
            exit(1);
          }
        } elsif ($ret != 0) {
          print STDERR "queue.pl: Warning: qstat command returned status $ret (qstat -j $sge_job_id,$!)\n";
        }
      }
    }
  }
  $all_syncfiles = join(" ", @syncfiles);
  system("rm $all_syncfiles 2>/dev/null");
}

# OK, at this point we are synced; we know the job is done.
# But we don't know about its exit status.  We'll look at $logfile for this.
# First work out an array @logfiles of file-locations we need to
# read (just one, unless it's an array job).
@logfiles = ();
if (!defined $jobname) { # not an array job.
  push @logfiles, $logfile;
} else {
  for ($jobid = $jobstart; $jobid <= $jobend; $jobid++) {
    $l = $logfile; 
    $l =~ s/\$SGE_TASK_ID/$jobid/g;
    push @logfiles, $l;
  }
}

$num_failed = 0;
foreach $l (@logfiles) {
  @wait_times = (0.1, 0.2, 0.2, 0.3, 0.5, 0.5, 1.0, 2.0, 5.0, 5.0, 5.0, 10.0, 25.0);
  for ($iter = 0; $iter <= @wait_times; $iter++) {
    $line = `tail -10 $l 2>/dev/null`; # Note: although this line should be the last
    # line of the file, I've seen cases where it was not quite the last line because
    # of delayed output by the process that was running, or processes it had called.
    # so tail -10 gives it a little leeway.
    if ($line =~ m/with status (\d+)/) {
      $status = $1;
      last;
    } else {
      if ($iter < @wait_times) {
        sleep($wait_times[$iter]);
      } else {
        if (! -f $l) {
          print STDERR "Log-file $l does not exist.\n";
        } else {
          print STDERR "The last line of log-file $l does not seem to indicate the "
            . "return status as expected\n";
        }
        exit(1);                # Something went wrong with the queue, or the
        # machine it was running on, probably.
      }
    }
  }
  # OK, now we have $status, which is the return-status of
  # the command in the job.
  if ($status != 0) { $num_failed++; }
}
if ($num_failed == 0) { exit(0); }
else { # we failed.
  if (@logfiles == 1) {
    if (defined $jobname) { $logfile =~ s/\$SGE_TASK_ID/$jobstart/g; }
    print STDERR "queue.pl: job failed with status $status, log is in $logfile\n";
    if ($logfile =~ m/JOB/) {
      print STDERR "queue.pl: probably you forgot to put JOB=1:\$nj in your script.\n";
    }
  } else {
    if (defined $jobname) { $logfile =~ s/\$SGE_TASK_ID/*/g; }
    $numjobs = 1 + $jobend - $jobstart;
    print STDERR "queue.pl: $num_failed / $numjobs failed, log is in $logfile\n";
  }
  exit(1);
}
