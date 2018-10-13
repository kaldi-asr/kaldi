#!/usr/bin/env perl

use strict;
use warnings;
use Carp;

BEGIN {
  @ARGV >= 9 or croak "
Usage: 
$0  <QSUB_ARGUMENTS>
where qsub arguments are:
<PROJECT_NAME>
<QUEUE_NAME>
<WALLTIME>
<NUMBER_OF_NODES>
<NUMBER_OF_Threads>
<PLACE>
<EXCLUSIVE>
<JOBSTART>
<JOBEND>
example: 
$0 ARLAP14877100  debug 00:30:00 8 40 scatter excl 1 10
";
}

use File::Basename;
use Cwd qw(getcwd);
my $cwd = getcwd;

# Start initializing variables: 
my $qsub_opts = "";
my $num_threads = 1;
my $gpu = 0;
my $config = "conf/pbspro.conf";
my %cli_options = ();
my $jobname = 'JOB';
my $jobstart = 0;
my $jobend = 0;
my $array_job = 1;
my $logfile = "";
my %cli_config_options = ();
my %cli_default_options = ();
my $opened_config_file = 1;
my $default_config_file = "conf/pbspro.conf";
my $qsub_cmd = "";
my $queue_scriptfile = "";
my $queue_name="debug";
my $cmd = "";
my $queue_array_opt = "";
my $ret = "";
my $tool = "";
my $walltime = "";
my $project = "";
my $nodes = "";
my $place = "";
my $exclusive = "";
my $dir = "";
my $base = "";
my $qdir = '';
# End initializing variables

($project,$queue_name,$walltime,$nodes,$num_threads,$place,$exclusive,$jobstart,$jobend,$cmd) = @ARGV;
my @cmd = @ARGV[9..$#ARGV];

$cmd = join ' ', @cmd;
my $cmd_string = join '_', @cmd;
$cmd_string =~ s:/:_:g;
$cmd_string =~ s:\.:_:g;

my $opciones = ' -a ' . $project . ' -q ' . $queue_name . ' -l walltime=' . $walltime . ' -l select=' . $nodes . ':ncpus=' . $num_threads . ' -l place=' . $place . ':' . $exclusive . ' -J ' . $jobstart . '-' . $jobend;

$qsub_opts .= "-V -S #!/bin/bash";
$qsub_opts .= $opciones;

#($qsub_opts,$num_threads,$array_job,$jobname,$jobstart,$jobend) = &process_arguments($config,$array_job,$jobname,$jobstart,$jobend);

$cwd = getcwd();

$logfile = "${cmd_string}.${jobname}.log";

$dir = dirname $logfile;

$base = basename $logfile, '.log';

$qdir = "$dir/q";

$qdir =~ s:/(log|LOG)/*q:/q:; # If qdir ends in .../log/q, make it just .../q.

my $queue_logfile = "$qdir/${base}.log";

if (!-d $dir) {
  system "mkdir -p $dir 2>/dev/null";
}

if (!-d $dir) {
  croak "Cannot make the directory $dir\n";
}

if (! -d "$qdir") {
  system "mkdir $qdir 2>/dev/null";
  sleep(5);
}

$logfile =~ s/$jobname/\$PBS_JOBID/g;

$queue_scriptfile = $queue_logfile;

$queue_scriptfile =~ s/.log$/.sh/;

if ($queue_scriptfile !~ m:^/:) {
  $queue_scriptfile = $cwd . "/" . $queue_scriptfile; # just in case.
}
$queue_logfile = $queue_scriptfile;
$queue_logfile =~ s/sh$/log/;
my $syncfile = "$qdir/done.$$";

#system "rm $queue_logfile $syncfile 2>/dev/null";

#unlink($queue_logfile, $syncfile);
&write_script($queue_scriptfile,$cwd,$cmd,$logfile,$num_threads,$syncfile,$qsub_cmd,$queue_logfile,$queue_array_opt,$array_job);

chmod 0755, $queue_scriptfile;my $pbs_job_id;

&submit_job($qsub_opts);

my @syncfiles = ();
for my $jobid ($jobstart..$jobend) {
  push @syncfiles, "$syncfile.$jobid";
}

{
  # Get the PBS job-id from the log file in q/
  open my $L, '<', $queue_logfile or warn "problem  with $queue_logfile $!";
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
    warn "Error: log file $queue_logfile does not specify the pbs job-id.";
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
          $logfile =~ s/\$PBS_JOBID/$job_id/g;
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
      } elsif ($ret != 0) {          print STDERR "pbspro.pl: Warning: qstat command returned status $ret (qstat -t $pbs_job_id,$!)\n";
      }
    }
  }
}
my $all_syncfiles = join(" ", @syncfiles);
system("rm $all_syncfiles 2>/dev/null");
my @logfiles = ();
my $l = "";
for my $jobid ($jobstart..$jobend) {
  $l = $logfile;
  $l =~ s/\$PBS_JOBID/$jobid/g;
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
    if (defined $jobname) { $logfile =~ s/\$PBS_JOBID/$jobstart/g; }
    print STDERR "pbspro.pl: job failed with status $status, log is in $logfile\n";
    if ($logfile =~ m/JOB/) {
      print STDERR "pbspro.pl: probably you forgot to put JOB=1:\$nj in your script.\n";
    }
  } else {
    if (defined $jobname) { $logfile =~ s/\$PBS_JOBID/*/g; }
    my $numjobs = 1 + $jobend - $jobstart;
    print STDERR "pbspro.pl: $num_failed / $numjobs failed, log is in $logfile\n";
  }
  exit(1);
}

sub write_script {
    my ($queue_scriptfile,$cwd,$cmd,$logfile,$num_threads,$syncfile,$qsub_cmd,$queue_logfile,$queue_array_opt,$array_job) = @_;
  open my $Q, '+>', $queue_scriptfile or croak "problems with $queue_scriptfile $!";
  print $Q "#!/bin/bash\n";
  print $Q "cd $cwd\n";
  print $Q ". ./path.sh\n";
  print $Q "( echo '#' Running on \`hostname\`\n";
  print $Q "  echo '#' Started at \`date\`\n";
  print $Q "  echo -n '# '; cat <<EOF\n";
  print $Q "$cmd\n";
  print $Q "EOF\n";
  print $Q ") >$logfile\n";
  print $Q "time1=\`date +\"%s\"\`\n";
  print$Q " ( $cmd ) 2>>$logfile >>$logfile\n";
 print $Q "ret=\$?\n";
  print $Q "time2=\`date +\"%s\"\`\n";
  print $Q "echo '#' Accounting: time=\$((\$time2-\$time1)) threads=$num_threads >>$logfile\n";
print $Q "echo '#' Finished at \`date\` with status \$ret >>$logfile\n";
  print $Q "[ \$ret -eq 137 ] && exit 100;\n"; # If process was killed (e.g. oom) it will exit with status 137;
  if ($array_job == 0) {
    print $Q "touch $syncfile\n";
  } else {
    print $Q "touch $syncfile.\$PBS_JOBID\n";
  }
  print $Q "exit \$[\$ret ? 1 : 0]\n"; # avoid status 100 which grid-engine
  print $Q "## submitted with:\n";       # treats specially.
  $qsub_cmd .= "-o $queue_logfile $qsub_opts $queue_array_opt $queue_scriptfile >>$queue_logfile 2>&1";
  print $Q "# $qsub_cmd\n";
  close $Q;
}


sub submit_job {
  my ($qsub_opts) = @_;
  for my $try (1..5) {
    my $ret = system "qsub $qsub_opts";
    if ($ret != 0) {
      print STDERR "pbspro.pl: Error submitting jobs to queue (return status was $ret)\nqsub $qsub_opts";
      print STDERR "queue log file is $queue_logfile, command was $qsub_cmd\n";
      my $err = `tail $queue_logfile`;
      print STDERR "Output of qsub was: $err\n";
      my $waitfor = 20;
      print STDERR "Trying again after $waitfor seconts\n";
      sleep $waitfor;
    } else {
      # break from the loop.
      last;
    }
  }
}
