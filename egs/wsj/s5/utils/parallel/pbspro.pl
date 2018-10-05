#!/usr/bin/env perl
use strict;
use warnings;
use Carp;

use File::Basename;

BEGIN {
    @ARGV == 5 or croak "USAGE: pbspro.pl <LOGFILE> <JJOBNAME> <JOBSTART> <JOBEND> <NUM_THREADS>
  example:
    $0 foo.log JOB 1 10 40
";
}

my ($logfile,$jobname,$jobstart,$jobend,$num_threads) = @ARGV;

my $qsub_cmd = "";
    my $queue_scriptfile = "";
my $cwd = `pwd`;

my $cmd = "";

foreach my $x (@ARGV) {
  if ($x =~ /^\S+$/) {
    $cmd .= $x . " ";
  } elsif ($x =~ m:\":) {
 $cmd .= "'$x' "; }
  else {
    $cmd .= "\"$x\" ";
  }
}

# Work out the location of the script file, and open it for writing.

my $dir = dirname $logfile;
my $base = basename $logfile;
my $qdir = "$dir/q";
$qdir =~ s:/(log|LOG)/*q:/q:; # If qdir ends in .../log/q, make it just .../q.
my $queue_logfile = "$qdir/$base";

if (!-d $dir) {
  # another job may be doing this...
  system "mkdir -p $dir 2>/dev/null";
}

if (!-d $dir) {
  croak "Cannot make the directory $dir\n";
}

if (! -d "$qdir") {
  system "mkdir $qdir 2>/dev/null";
  sleep(5);
}

my $queue_array_opt = "";

$queue_array_opt = "-J $jobstart-$jobend";
$logfile =~ s/$jobname/\$PBS_ARRAY_INDEX/g;
$cmd =~ s/$jobname/\$\{PBS_ARRAY_INDEX\}/g;
$queue_logfile =~ s/\.?$jobname//;

$queue_scriptfile = $queue_logfile;
($queue_scriptfile =~ s/\.[a-zA-Z]{1,5}$/.sh/) || ($queue_scriptfile .= ".sh");
if ($queue_scriptfile !~ m:^/:) {
  $queue_scriptfile = $cwd . "/" . $queue_scriptfile;
  # just in case.
}

my $syncfile = "$qdir/done.$$";

system("rm $queue_logfile $syncfile 2>/dev/null");

open my $Q, '+>', $queue_scriptfile or croak "problems with $queue_scriptfile $!";

print $Q "#!/bin/bash\n";
print $Q "cd $cwd\n";
print $Q ". ./path.sh\n";
print $Q "( echo '#' Running on \`hostname\`\n";
print $Q "  echo '#' Started at \`date\`\n";
print $Q "  echo -n '# '; cat <<EOF\n";
print $Q "$cmd\n"; # this is a way of echoing the command into a comment in the log file,
print $Q "EOF\n"; # without having to escape things like "|" and quote characters.
print $Q ") >$logfile\n";
print $Q "time1=\`date +\"%s\"\`\n";
  print$Q " ( $cmd ) 2>>$logfile >>$logfile\n";
print $Q "ret=\$?\n";
print $Q "time2=\`date +\"%s\"\`\n";
print $Q "echo '#' Accounting: time=\$((\$time2-\$time1)) threads=$num_threads >>$logfile\n";
print $Q "echo '#' Finished at \`date\` with status \$ret >>$logfile\n";
print $Q "[ \$ret -eq 137 ] && exit 100;\n"; # If process was killed (e.g. oom) it will exit with status 137;

# touch a bunch of sync-files.
print $Q "touch $syncfile.\$PBS_ARRAY_INDEX\n";

print $Q "exit \$[\$ret ? 1 : 0]\n"; # avoid status 100 which grid-engine
print $Q "## submitted with:\n";       # treats specially.
$qsub_cmd .= "-o $queue_logfile $qsub_opts $queue_array_opt $queue_scriptfile >>$queue_logfile 2>&1";
print $Q "# $qsub_cmd\n";
close $Q;

my $ret = system ($qsub_cmd);
if ($ret != 0) {
  print STDERR "queue.pl: error submitting jobs to queue (return status was $ret)\n";
  print STDERR "queue log file is $queue_logfile, command was $qsub_cmd\n";
  print STDERR `tail $queue_logfile`;
  exit(1);
}

my $pbs_job_id;
my @syncfiles = ();
for my $jobid ($jobstart..$jobend) {
  push @syncfiles, "$syncfile.$jobid";
}

{
  # Get the PBS job-id from the log file in q/
  open my $L, '<', $queue_logfile or croak "problem  with$queue_logfile $!";
  undef $pbs_job_id;
  while ( my $line = <$L> ) {
    chomp $line;
    if ( $line =~ /(\d+.+pbsserver)/) {
      if (defined $pbs_job_id) {
        croak "Error: your job was submitted more than once (see $queue_logfile)";
      } else {
        $pbs_job_id = $1;
      }
    }
  }
  close $L;
  if (!defined $pbs_job_id) {
    croak "Error: log file $queue_logfile does not specify the pbs job-id.";
  }
}
my $check_pbs_job_ctr=1;

my $wait = 0.1;
my $counter = 0;
foreach my $f (@syncfiles) {
  while (! -f $f) {
    sleep($wait);
    $wait *= 1.2;
    if ($wait > 3.0) {
      $wait = 3.0;
      if (rand() < 0.25) {
        if (rand() > 0.5) {
          system("touch $qdir/.kick");
        } else {
          system("rm $qdir/.kick 2>/dev/null");
        }
      }
      if ($counter++ % 10 == 0) {
        system("ls $qdir >/dev/null");
      }
    }

    if (($check_pbs_job_ctr++ % 10) == 0) {
      if ( -f $f ) { next; };
      $ret = system("qstat -t $pbs_job_id >/dev/null 2>/dev/null");
      if ($ret>>8 == 1) {
        sleep(3);
        system("touch $qdir/.kick");
        system("rm $qdir/.kick 2>/dev/null");
        if ( -f $f ) { next; }
        sleep(7);
        system("touch $qdir/.kick");
        sleep(1);
        system("rm $qdir/.kick 2>/dev/null");
        if ( -f $f ) {  next; }
        sleep(60);
        system("touch $qdir/.kick");
        sleep(1);
        system("rm $qdir/.kick 2>/dev/null");
        if ( -f $f ) { next; }
        $f =~ m/\.(\d+)$/ || croak "Bad sync-file name $f";
        my $job_id = $1;
        if (defined $jobname) {
          $logfile =~ s/\$PBS_ARRAY_INDEX/$job_id/g;
        }
        my $last_line = `tail -n 1 $logfile`;
        if ($last_line =~ m/status 0$/ && (-M $logfile) < 0) {
          print STDERR "**pbspro.pl: syncfile $f was not created but job seems\n" .
            "**to have finished OK.  Probably your file-system has problems.\n" .
            "**This is just a warning.\n";
	  last;
        } else {
          chop $last_line;
          print STDERR "pbspro.pl: Error, unfinished job no " .
            "longer exists, log is in $logfile, last line is '$last_line', " .
            "syncfile is $f, return status of qstat was $ret\n" .
            "Possible reasons: a) Exceeded time limit? -> Use more jobs!" .
            " b) Shutdown/Frozen machine? -> Run again!\n";
	  exit(1);
        }
      } elsif ($ret != 0) {          print STDERR "pbspro.pl: Warning: qstat command returned status $ret (qstat -t $sge_job_id,$!)\n";
      }
    }
  }
}
my $all_syncfiles = join(" ", @syncfiles);
system("rm $all_syncfiles 2>/dev/null");
my @logfiles = ();
for my $jobid ($jobstart..$jobend;) {
  my $l = $logfile;
  $l =~ s/\$PBS_ARRAY_INDEX/$jobid/g;
  push @logfiles, $l;
}

my $num_failed = 0;
my $status = 1;
foreach my $l (@logfiles) {
  my @wait_times = (0.1, 0.2, 0.2, 0.3, 0.5, 0.5, 1.0, 2.0, 5.0, 5.0, 5.0, 10.0, 25.0);
  for (my $iter = 0; $iter <= @wait_times; $iter++) {
    my $line = `tail -10 $l 2>/dev/null`;
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
        exit(1);
      }
    }
  }
  if ($status != 0) { $num_failed++; }
}
if ($num_failed == 0) { exit(0); }
else { # we failed.
  if (@logfiles == 1) {
    if (defined $jobname) { $logfile =~ s/\$PBS_ARRAY_INDEX/$jobstart/g; }
    print STDERR "pbspro.pl: job failed with status $status, log is in $logfile\n";
    if ($logfile =~ m/JOB/) {
      print STDERR "pbspro.pl: probably you forgot to put JOB=1:\$nj in your script.\n";
    }
  } else {
    if (defined $jobname) { $logfile =~ s/\$PBS_ARRAY_INDEX/*/g; }
    my $numjobs = 1 + $jobend - $jobstart;
    print STDERR "pbspro.pl: $num_failed / $numjobs failed, log is in $logfile\n";
  }
  exit(1);
}
