#!/usr/bin/env perl

use strict;
use warnings;
use Carp;

BEGIN {
  @ARGV >= 2 or croak "Usage: pbspro.pl [options] [JOB=1:n] log-file command-line arguments...\n" .
    "example: \n$0 foo.log echo baz\n" .
    " (which will echo \"baz\", with stdout and stderr directed to foo.log)\n" .
    "or: pbspro.pl -q all.q\@xyz foo.log echo bar \| sed s/bar/baz/ \n" .
    " (which is an example of using a pipe; you can provide other escaped bash constructs)\n" .
    "or: pbspro.pl -q all.q\@qyz JOB=1:10 foo.JOB.log echo JOB \n" .
    " (which illustrates the mechanism to submit parallel jobs;" .
    "It uses qstat to work out when the job finished\n" .
    "Options:\n" .
    "  --config <config-file> (default: conf/pbspro.conf)\n" .
    "  --mem <mem-requirement> (e.g. --mem 2G, --mem 500M, \n" .   "                           also support K and numbers mean bytes)\n" .    "  --num-threads <num-threads> (default: 1)\n" .
    "  --max-jobs-run <num-jobs>\n" .
       "  --gpu <0|1> (default: 0)\n
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
my $jobname = '';
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
my $cmd = "";
my $queue_array_opt = "";
my $ret = "";
# End initializing variables

foreach my $x (@ARGV) {
  # If string contains no spaces, take as-is.
  if ($x =~ /^\S+$/) {
    $cmd .= $x . " ";
  } elsif ($x =~ /\"/) {
    # else if no dbl-quotes, use single
    $cmd .= "'$x' ";
  } else {
    # else use double.
    $cmd .= "\"$x\" ";
  }
}

($qsub_opts,$num_threads,$array_job,$jobname,$jobstart,$jobend) = &process_arguments($config,$array_job,$jobname,$jobstart,$jobend);

$cwd = getcwd();
$logfile = shift @ARGV;

if ($array_job == 1 && $logfile !~ m/$jobname/
  && $jobend > $jobstart) {
  print STDERR "pbspro.pl: you are trying to run a parallel job but "
  . "you are putting the output into just one log file ($logfile)\n";
  exit(1);
}

my $dir = dirname $logfile;
my $base = basename $logfile;
my $qdir = "$dir/q";
$qdir =~ s:/(log|LOG)/*q:/q:; # If qdir ends in .../log/q, make it just .../q.

my $queue_logfile = "$qdir/$base";
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

if ($array_job == 1) {
  $queue_array_opt = "-J $jobstart-$jobend";
  $logfile =~ s/$jobname/\$PBS_JOBID/g;
  $cmd =~ s/$jobname/\$\{PBS_JOBID\}/g;
  $queue_logfile =~ s/\.?$jobname=//;
  $queue_logfile =~ s/\://;
}

$queue_scriptfile = $queue_logfile;
($queue_scriptfile =~ s/\.[a-zA-Z]{1,5}$/.sh/) || ($queue_scriptfile .= ".sh");

if ($queue_scriptfile !~ m:^/:) {
  $queue_scriptfile = $cwd . "/" . $queue_scriptfile; # just in case.
}

my $syncfile = "$qdir/done.$$";

#system "rm $queue_logfile $syncfile 2>/dev/null";

#unlink($queue_logfile, $syncfile);
&write_script($queue_scriptfile,$cwd,$cmd,$logfile,$num_threads,$syncfile,$qsub_cmd,$queue_logfile,$queue_array_opt,$array_job);

chmod 0755, $queue_scriptfile;my $pbs_job_id;
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
sub process_arguments {
    my ($config,$array_job,$jobname,$jobstart,$jobend) = @_;
    my @out = ();
  #  allow the JOB=1:n option to be interleaved with the options to qsub.
  for my $x (1..2) {
    while (@ARGV >= 2 && $ARGV[0] =~ m:^-:) {
      my $switch = shift @ARGV;
      if ($switch eq "-V") {
        $qsub_opts .= "-V ";
      } else {
        my $argument = shift @ARGV;
        if ($argument =~ m/^--/) {
          print STDERR "pbspro.pl: Warning: suspicious argument '$argument' to $switch; starts with '-'\n";
        }
        if ($switch eq "-pe") { # e.g. -pe smp 5
          my $argument2 = shift @ARGV;
          $qsub_opts .= "$switch $argument $argument2 ";
          $num_threads = $argument2;
        } elsif ($switch =~ m/^--/) { # Config options
          # Convert CLI option to variable name
          # by removing '--' from the switch and replacing any
          # '-' with a '_'
          $switch =~ s/^--//;
          $switch =~ s/-/_/g;
          $cli_options{$switch} = $argument;
        } else {
          # Other qsub options - passed as is
          $qsub_opts .= "$switch $argument ";
        }
      }
    }
    if ($ARGV[0] =~ m/^([\w_][\w\d_]*)+=(\d+):(\d+)$/) { # e.g. JOB=1:20
      $array_job = 1;
      $jobname = $1;
      $jobstart = $2;
      $jobend = $3;
      shift;
      if ($jobstart > $jobend) {
        croak "pbspro.pl: invalid job range $ARGV[0]";
      }
      if ($jobstart <= 0) {
	  croak "pbspro: invalid job range $ARGV[0], start must be strictly positive.";
      }
    } elsif ($ARGV[0] =~ m/^([\w_][\w\d_]*)+=(\d+)$/) { # e.g. JOB=1.
      $array_job = 1;
      $jobname = $1;
      $jobstart = $2;
      $jobend = $2;
      shift;
    } elsif ($ARGV[0] =~ m/.+\=.*\:.*$/) {
      print STDERR "pbspro.pl: Warning: suspicious first argument to queue.pl: $ARGV[0]\n";
    }
  }

  ($config,$default_config_file) = &process_config_file($config);

  ($qsub_opts,$config) = &process_cli(\%cli_options);
  push @out, $qsub_opts,$num_threads,$array_job,$jobname,$jobstart,$jobend;
  return@out; 
}

sub process_config_file {
  my ($config) = @_;
  # Here the configuration options specified by the user on the command line
  # (e.g. --mem 2G) are converted to options to the qsub system as defined in
  # the config file. (e.g. if the config file has the line
  # "option mem=* -l ram_free=$0,mem_free=$0"
  # and the user has specified '--mem 2G' on the command line, the options
  # passed to queue system would be "-l ram_free=2G,mem_free=2G

  open my $CONFIG, '<', $config or $opened_config_file = 0;

  if ($opened_config_file == 0 && exists($cli_options{"config"})) {
    print STDERR "Could not open config file $config\n";
    exit(1);
  } elsif ($opened_config_file == 0 && !exists($cli_options{"config"})) {
    # Open the default config file instead
    open  my $CONFIG, "echo '$default_config_file' |" or croak "Unable to open pipe $!\n";
    $config = "Default config";
  }

  my $qsub_cmd = "";
  my $read_command = 0;

  LINE: while( my $line = <$CONFIG>) {
    my $linea = $line;
    chomp $line;
    $line =~ s/\s*#.*//g;
    next LINE if ( $line eq "");
    if ( $line =~ /^command (.+)/ ) {
      $read_command = 1;
      $qsub_cmd = $1 . " ";
    } elsif ( $line =~ /^option ([^=]+)=\* (.+)$/ ) {
      # Config option that needs replacement with parameter value read from CLI
      # e.g.: option mem=* -l mem_free=$0,ram_free=$0
      # mem
      my $option = $1;
      # -l mem_free=$0,ram_free=$0
      my $arg= $2;
      if ( $arg !~ m:\$0: ) {
        croak "pbspro.pl: Unable to parse line '$linea' in config file ($config)\n";
      }
      if ( exists $cli_options{$option} ) {
        # Replace $0 with the argument read from command line.
        # e.g. "-l mem_free=$0,ram_free=$0" -> "-l mem_free=2G,ram_free=2G"
        $arg =~ s/\$0/$cli_options{$option}/g;
        $cli_config_options{$option} = $arg;
      }
    } elsif ( $line =~ /^option ([^=]+)=(\S+)\s?(.*)$/ ) {
      # Config option that does not need replacement
      # e.g. option gpu=0 -q all.q
      # gpu
      my $option = $1;
      # 0
      my $value = $2;
      # -q all.q
      my $arg = $3;
      if ( exists $cli_options{$option} ) {
        $cli_default_options{($option,$value)} = $arg;
      }
    } elsif ( $line =~ /^default (\S+)=(\S+)/ ) {
      # Default options. Used for setting default values to options i.e. when
      # the user does not specify the option on the command line
      # e.g. default gpu=0
      # gpu
      my $option = $1;
       # 0
      my $value = $2;
      if ( ! exists $cli_options{$option} ) {
        # If the user has specified this option on the command line, then we
        # don't have to do anything
        $cli_options{$option} = $value;
      }
    } else {
      print STDERR "pbspro.pl: unable to parse line '$linea' in config file ($config)\n";
      exit(1);
    }
  }

  close $CONFIG;

  if ($read_command != 1) {
    print STDERR "pbspro.pl: config file ($config) does not contain the line \"command .*\"\n";
    exit(1);
  }
  if (exists $cli_options{"config"}) {
    $config = $cli_options{"config"};
  }

my $default_config_file = <<'EOF';
# Default configuration
command qsub -V -v PATH -S /bin/bash -l mem=4G
option mem=* -l mem=$0
option mem=0          # Do not add anything to qsub_opts
option num_threads=* -l ncpus=$0
option num_threads=1  # Do not add anything to qsub_opts
default gpu=0
option gpu=0
option gpu=* -l ncpus=$0
EOF
  return ($config,$default_config_file);
}

sub process_cli {
  my ($cli_options) = @_;
  OPTION: for my $option (keys %{$cli_options}) {
    next OPTION if ( $option eq "config");
    next OPTION if ( $option eq "max_jobs_run" && $array_job != 1 );
    my $value = $cli_options{$option};

    if (exists $cli_default_options{($option,$value)}) {
      $qsub_opts .= "$cli_default_options{($option,$value)} ";
    } elsif (exists $cli_config_options{$option}) {
      $qsub_opts .= "$cli_config_options{$option} ";
    } else {
      if ( $opened_config_file == 0) {
        $config = "default config file";
      }
      croak "pbspro.pl: Command line option $option not described in $config (or value '$value' not allowed)\n";
    }
  }
  return ($qsub_opts,$config);
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
  my ($qsub_cmd) = @_;
  for my $try (1..5) {
    my $ret = system $qsub_cmd;
    if ($ret != 0) {
      print STDERR "pbspro.pl: Error submitting jobs to queue (return status was $ret)\n";
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
