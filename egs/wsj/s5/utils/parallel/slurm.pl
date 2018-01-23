#!/usr/bin/env perl
use strict;
use warnings;

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).
#           2014  Vimal Manohar (Johns Hopkins University)
#           2015  Johns Hopkins University (Yenda Trmal <jtrmal@gmail.com>>)
# Apache 2.0.

use File::Basename;
use Cwd;
use Getopt::Long;

# slurm.pl was created from the queue.pl
# queue.pl has the same functionality as run.pl, except that
# it runs the job in question on the queue (Sun GridEngine).
# This version of queue.pl uses the task array functionality
# of the grid engine.  Note: it's different from the queue.pl
# in the s4 and earlier scripts.

# The script now supports configuring the queue system using a config file
# (default in conf/queue.conf; but can be passed specified with --config option)
# and a set of command line options.
# The current script handles:
# 1) Normal configuration arguments
# For e.g. a command line option of "--gpu 1" could be converted into the option
# "-q g.q -l gpu=1" to qsub. How the CLI option is handled is determined by a
# line in the config file like
# gpu=* -q g.q -l gpu=$0
# $0 here in the line is replaced with the argument read from the CLI and the
# resulting string is passed to qsub.
# 2) Special arguments to options such as
# gpu=0
# If --gpu 0 is given in the command line, then no special "-q" is given.
# 3) Default argument
# default gpu=0
# If --gpu option is not passed in the command line, then the script behaves as
# if --gpu 0 was passed since 0 is specified as the default argument for that
# option
# 4) Arbitrary options and arguments.
# Any command line option starting with '--' and its argument would be handled
# as long as its defined in the config file.
# 5) Default behavior
# If the config file that is passed using is not readable, then the script
# behaves as if the queue has the following config file:
# $ cat conf/queue.conf
# # Default configuration
# command sbatch --export=PATH  -S /bin/bash -j y -l arch=*64*
# option mem=* --mem-per-cpu $0
# option mem=0          # Do not add anything to qsub_opts
# option num_threads=* --cpus-per-task $0
# option num_threads=1  # Do not add anything to qsub_opts
# option max_jobs_run=* -tc $0
# default gpu=0
# option gpu=0 -p shared
# option gpu=*  -p gpu  #this has to be figured out

#print STDERR "$0 " . join(" ", @ARGV) . "\n";

my $qsub_opts = "";
my $sync = 0;
my $num_threads = 1;
my $max_jobs_run;
my $gpu = 0;

my $config = "conf/slurm.conf";

my %cli_options = ();

my $jobname;
my $jobstart;
my $jobend;

my $array_job = 0;

sub print_usage() {
  print STDERR
   "Usage: $0 [options] [JOB=1:n] log-file command-line arguments...\n" .
   "e.g.: $0 foo.log echo baz\n" .
   " (which will echo \"baz\", with stdout and stderr directed to foo.log)\n" .
   "or: $0 -q all.q\@xyz foo.log echo bar \| sed s/bar/baz/ \n" .
   " (which is an example of using a pipe; you can provide other escaped bash constructs)\n" .
   "or: $0 -q all.q\@qyz JOB=1:10 foo.JOB.log echo JOB \n" .
   " (which illustrates the mechanism to submit parallel jobs; note, you can use \n" .
   "  another string other than JOB)\n" .
   "Note: if you pass the \"-sync y\" option to qsub, this script will take note\n" .
   "and change its behavior.  Otherwise it uses squeue to work out when the job finished\n" .
   "Options:\n" .
   "  --config <config-file> (default: $config)\n" .
   "  --mem <mem-requirement> (e.g. --mem 2G, --mem 500M, \n" .
   "                           also support K and numbers mean bytes)\n" .
   "  --num-threads <num-threads> (default: $num_threads)\n" .
   "  --max-jobs-run <num-jobs>\n" .
   "  --gpu <0|1> (default: $gpu)\n";
  exit 1;
}

sub exec_command {
  # Execute command and return a tuple of stdout and exit code
  my $command = join ' ', @_;
  # To get the actual exit value, shift right by eight bits.
  ($_ = `$command 2>&1`, $? >> 8);
}

if (@ARGV < 2) {
  print_usage();
}

for (my $x = 1; $x <= 3; $x++) { # This for-loop is to
  # allow the JOB=1:n option to be interleaved with the
  # options to qsub.
  while (@ARGV >= 2 && $ARGV[0] =~ m:^-:) {
    my $switch = shift @ARGV;

    if ($switch eq "-V") {
      $qsub_opts .= "-V ";
    } else {
      my $argument = shift @ARGV;
      if ($argument =~ m/^--/) {
        print STDERR "WARNING: suspicious argument '$argument' to $switch; starts with '-'\n";
      }
      if ($switch eq "-sync" && $argument =~ m/^[yY]/) {
        $sync = 1;
        $qsub_opts .= "$switch $argument ";
      } elsif ($switch eq "-pe") { # e.g. -pe smp 5
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
      } else {  # Other qsub options - passed as is
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
      die "$0: invalid job range $ARGV[0]";
    }
    if ($jobstart <= 0) {
      die "$0: invalid job range $ARGV[0], start must be strictly positive (this is a GridEngine limitation).";
    }
  } elsif ($ARGV[0] =~ m/^([\w_][\w\d_]*)+=(\d+)$/) { # e.g. JOB=1.
    $array_job = 1;
    $jobname = $1;
    $jobstart = $2;
    $jobend = $2;
    shift;
  } elsif ($ARGV[0] =~ m/.+\=.*\:.*$/) {
    print STDERR "Warning: suspicious first argument to $0: $ARGV[0]\n";
  }
}

if (@ARGV < 2) {
  print_usage();
}

if (exists $cli_options{"config"}) {
  $config = $cli_options{"config"};
}

my $default_config_file = <<'EOF';
# Default configuration
command sbatch --export=PATH  --ntasks-per-node=1
option time=* --time $0
option mem=* --mem-per-cpu $0
option mem=0          # Do not add anything to qsub_opts
option num_threads=* --cpus-per-task $0 --ntasks-per-node=1
option num_threads=1 --cpus-per-task 1  --ntasks-per-node=1 # Do not add anything to qsub_opts
default gpu=0
option gpu=0 -p shared
option gpu=* -p gpu --gres=gpu:$0 --time 4:0:0  # this has to be figured out
# note: the --max-jobs-run option is supported as a special case
# by slurm.pl and you don't have to handle it in the config file.
EOF

# Here the configuration options specified by the user on the command line
# (e.g. --mem 2G) are converted to options to the qsub system as defined in
# the config file. (e.g. if the config file has the line
# "option mem=* -l ram_free=$0,mem_free=$0"
# and the user has specified '--mem 2G' on the command line, the options
# passed to queue system would be "-l ram_free=2G,mem_free=2G
# A more detailed description of the ways the options would be handled is at
# the top of this file.

my $opened_config_file = 1;

open CONFIG, "<$config" or $opened_config_file = 0;

my %cli_config_options = ();
my %cli_default_options = ();

if ($opened_config_file == 0 && exists($cli_options{"config"})) {
  print STDERR "Could not open config file $config\n";
  exit(1);
} elsif ($opened_config_file == 0 && !exists($cli_options{"config"})) {
  # Open the default config file instead
  open (CONFIG, "echo '$default_config_file' |") or die "Unable to open pipe\n";
  $config = "Default config";
}

my $qsub_cmd = "";
my $read_command = 0;

while(<CONFIG>) {
  chomp;
  my $line = $_;
  $_ =~ s/\s*#.*//g;
  if ($_ eq "") { next; }
  if ($_ =~ /^command (.+)/) {
    $read_command = 1;
    $qsub_cmd = $1 . " ";
  } elsif ($_ =~ m/^option ([^=]+)=\* (.+)$/) {
    # Config option that needs replacement with parameter value read from CLI
    # e.g.: option mem=* -l mem_free=$0,ram_free=$0
    my $option = $1;     # mem
    my $arg= $2;         # -l mem_free=$0,ram_free=$0
    if ($arg !~ m:\$0:) {
      print STDERR "Warning: the line '$line' in config file ($config) does not substitution variable \$0\n";
    }
    if (exists $cli_options{$option}) {
      # Replace $0 with the argument read from command line.
      # e.g. "-l mem_free=$0,ram_free=$0" -> "-l mem_free=2G,ram_free=2G"
      $arg =~ s/\$0/$cli_options{$option}/g;
      $cli_config_options{$option} = $arg;
    }
  } elsif ($_ =~ m/^option ([^=]+)=(\S+)\s?(.*)$/) {
    # Config option that does not need replacement
    # e.g. option gpu=0 -q all.q
    my $option = $1;      # gpu
    my $value = $2;       # 0
    my $arg = $3;         # -q all.q
    if (exists $cli_options{$option}) {
      $cli_default_options{($option,$value)} = $arg;
    }
  } elsif ($_ =~ m/^default (\S+)=(\S+)/) {
    # Default options. Used for setting default values to options i.e. when
    # the user does not specify the option on the command line
    # e.g. default gpu=0
    my $option = $1;  # gpu
    my $value = $2;   # 0
    if (!exists $cli_options{$option}) {
      # If the user has specified this option on the command line, then we
      # don't have to do anything
      $cli_options{$option} = $value;
    }
  } else {
    print STDERR "$0: unable to parse line '$line' in config file ($config)\n";
    exit(1);
  }
}

close(CONFIG);

if ($read_command != 1) {
  print STDERR "$0: config file ($config) does not contain the line \"command .*\"\n";
  exit(1);
}

for my $option (keys %cli_options) {
  if ($option eq "config") { next; }

  my $value = $cli_options{$option};

  if ($option eq "max_jobs_run") {
    if ($array_job != 1) {
      print STDERR "Ignoring $option since this is not an array task.";
    } else {
      $max_jobs_run = $value;
    }
  } elsif (exists $cli_default_options{($option,$value)}) {
    $qsub_opts .= "$cli_default_options{($option,$value)} ";
  } elsif (exists $cli_config_options{$option}) {
    $qsub_opts .= "$cli_config_options{$option} ";
  } elsif (exists $cli_default_options{($option,"*")}) {
    $qsub_opts .= $cli_default_options{($option,"*")} . " ";
  } else {
    if ($opened_config_file == 0) {
      $config = "default config file";
    }
    die "$0: Command line option $option not described in $config (or value '$value' not allowed)\n";
  }
}

my $cwd = getcwd();
my $logfile = shift @ARGV;

if ($array_job == 1 && $logfile !~ m/$jobname/
    && $jobend > $jobstart) {
  print STDERR "$0: you are trying to run a parallel job but "
    . "you are putting the output into just one log file ($logfile)\n";
  exit(1);
}

#
# Work out the command; quote escaping is done here.
# Note: the rules for escaping stuff are worked out pretty
# arbitrarily, based on what we want it to do.  Some things that
# we pass as arguments to $0, such as "|", we want to be
# interpreted by bash, so we don't escape them.  Other things,
# such as archive specifiers like 'ark:gunzip -c foo.gz|', we want
# to be passed, in quotes, to the Kaldi program.  Our heuristic
# is that stuff with spaces in should be quoted.  This doesn't
# always work.
#
my $cmd = "";

foreach my $x (@ARGV) {
  if ($x =~ m/^\S+$/) { $cmd .= $x . " "; } # If string contains no spaces, take
                                            # as-is.
  elsif ($x =~ m:\":) { $cmd .= "'$x' "; } # else if no dbl-quotes, use single
  else { $cmd .= "\"$x\" "; }  # else use double.
}

#
# Work out the location of the script file, and open it for writing.
#
my $dir = dirname($logfile);
my $base = basename($logfile);
my $qdir = "$dir/q";
$qdir =~ s:/(log|LOG)/*q:/q:; # If qdir ends in .../log/q, make it just .../q.
my $queue_logfile = "$qdir/$base";

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

my $queue_array_opt = "";
if ($array_job == 1) { # It's an array job.
  if ($max_jobs_run) {
      $queue_array_opt = "--array ${jobstart}-${jobend}%${max_jobs_run}";
  } else {
      $queue_array_opt = "--array ${jobstart}-${jobend}";
  }
  $logfile =~ s/$jobname/\$SLURM_ARRAY_TASK_ID/g; # This variable will get
  # replaced by qsub, in each job, with the job-id.
  $cmd =~ s/$jobname/\$\{SLURM_ARRAY_TASK_ID\}/g; # same for the command...
  $queue_logfile =~ s/\.?$jobname//; # the log file in the q/ subdirectory
  # is for the queue to put its log, and this doesn't need the task array subscript
  # so we remove it.
}

# queue_scriptfile is as $queue_logfile [e.g. dir/q/foo.log] but
# with the suffix .sh.
my $queue_scriptfile = $queue_logfile;
($queue_scriptfile =~ s/\.[a-zA-Z]{1,5}$/.sh/) || ($queue_scriptfile .= ".sh");
if ($queue_scriptfile !~ m:^/:) {
  $queue_scriptfile = $cwd . "/" . $queue_scriptfile; # just in case.
}

# We'll write to the standard input of "qsub" (the file-handle Q),
# the job that we want it to execute.
# Also keep our current PATH around, just in case there was something
# in it that we need (although we also source ./path.sh)

my $syncfile = "$qdir/done.$$";

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
print Q "  set | grep SLURM | while read line; do echo \"# \$line\"; done\n";
print Q "  echo -n '# '; cat <<EOF\n";
print Q "$cmd\n"; # this is a way of echoing the command into a comment in the log file,
print Q "EOF\n"; # without having to escape things like "|" and quote characters.
print Q ") >$logfile\n";
print Q "if [ \"\$CUDA_VISIBLE_DEVICES\" == \"NoDevFiles\" ]; then\n";
print Q "  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... \n";
print Q "  )>>$logfile\n";
print Q "  unset CUDA_VISIBLE_DEVICES\n";
print Q "fi\n";
print Q "time1=\`date +\"%s\"\`\n";
print Q " ( $cmd ) &>>$logfile\n";
print Q "ret=\$?\n";
print Q "sync || true\n";
print Q "time2=\`date +\"%s\"\`\n";
print Q "echo '#' Accounting: begin_time=\$time1 >>$logfile\n";
print Q "echo '#' Accounting: end_time=\$time2 >>$logfile\n";
print Q "echo '#' Accounting: time=\$((\$time2-\$time1)) threads=$num_threads >>$logfile\n";
print Q "echo '#' Finished at \`date\` with status \$ret >>$logfile\n";
print Q "[ \$ret -eq 137 ] && exit 100;\n"; # If process was killed (e.g. oom) it will exit with status 137;
  # let the script return with status 100 which will put it to E state; more easily rerunnable.
if ($array_job == 0) { # not an array job
  print Q "touch $syncfile\n"; # so we know it's done.
} else {
  print Q "touch $syncfile.\$SLURM_ARRAY_TASK_ID\n"; # touch a bunch of sync-files.
}
print Q "exit \$[\$ret ? 1 : 0]\n"; # avoid status 100 which grid-engine
print Q "## submitted with:\n";       # treats specially.
$qsub_cmd .= " $qsub_opts --open-mode=append -e ${queue_logfile} -o ${queue_logfile} $queue_array_opt $queue_scriptfile >>$queue_logfile 2>&1";
print Q "# $qsub_cmd\n";
if (!close(Q)) { # close was not successful... || die "Could not close script file $shfile";
  die "Failed to close the script file (full disk?)";
}

my $ret = system ($qsub_cmd);
if ($ret != 0) {
  if ($sync && $ret == 256) { # this is the exit status when a job failed (bad exit status)
    if (defined $jobname) { $logfile =~ s/\$SLURM_ARRAY_TASK_ID/*/g; }
    print STDERR "$0: job writing to $logfile failed\n";
  } else {
    print STDERR "$0: error submitting jobs to queue (return status was $ret)\n";
    print STDERR "queue log file is $queue_logfile, command was $qsub_cmd\n";
    print STDERR `tail $queue_logfile`;
  }
  exit(1);
}

my $sge_job_id;
if (! $sync) { # We're not submitting with -sync y, so we
  # need to wait for the jobs to finish.  We wait for the
  # sync-files we "touched" in the script to exist.
  my @syncfiles = ();
  if (!defined $jobname) { # not an array job.
    push @syncfiles, $syncfile;
  } else {
    for (my $jobid = $jobstart; $jobid <= $jobend; $jobid++) {
      push @syncfiles, "$syncfile.$jobid";
    }
  }
  # We will need the sge_job_id, to check that job still exists
  { # Get the SLURM job-id from the log file in q/
    open(L, "<$queue_logfile") || die "Error opening log file $queue_logfile";
    undef $sge_job_id;
    while (<L>) {
      if (m/Submitted batch job (\d+)/) {
        if (defined $sge_job_id) {
          die "Error: your job was submitted more than once (see $queue_logfile)";
        } else {
          $sge_job_id = $1;
        }
      }
    }
    close(L);
    if (!defined $sge_job_id) {
      die "Error: log file $queue_logfile does not specify the SLURM job-id.";
    }
  }
  my $check_sge_job_ctr=1;
  #
  my $wait = 0.1;
  my $counter = 0;
  foreach my $f (@syncfiles) {
    # wait for them to finish one by one.
    while (! -f $f) {
      sleep($wait);
      $wait *= 1.2;
      if ($wait > 3.0) {
        $wait = 3.0; # never wait more than 3 seconds.
        # the following (.kick) commands are basically workarounds for NFS bugs.
        if (rand() < 0.25) { # don't do this every time...
          if (rand() > 0.5) {
            system("touch $qdir/.kick");
          } else {
            system("rm $qdir/.kick 2>/dev/null");
          }
        }
        if ($counter++ % 10 == 0) {
          # This seems to kick NFS in the teeth to cause it to refresh the
          # directory.  I've seen cases where it would indefinitely fail to get
          # updated, even though the file exists on the server.
          # Only do this every 10 waits (every 30 seconds) though, or if there
          # are many jobs waiting they can overwhelm the file server.
          system("ls $qdir >/dev/null");
        }
      }

      # Check that the job exists in SLURM. Job can be killed if duration
      # exceeds some hard limit, or in case of a machine shutdown.
      if (($check_sge_job_ctr++ % 10) == 0) { # Don't run qstat too often, avoid stress on SGE.
        if ( -f $f ) { next; }; #syncfile appeared: OK.
        # system(...) : To get the actual exit value, shift $ret right by eight bits.
        my ($squeue_output, $squeue_status) = exec_command("squeue -j $sge_job_id");
        if ($squeue_status == 1) {
          # Don't consider immediately missing job as error, first wait some
          sleep(4);
          ($squeue_output, $squeue_status) = exec_command("squeue -j $sge_job_id");
        }
        if ($squeue_status == 1) {
          # time to make sure it is not just delayed creation of the syncfile.

          # Don't consider immediately missing job as error, first wait some
          # time to make sure it is not just delayed creation of the syncfile.
          sleep(4);
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
          my $job_id = $1;
          if (defined $jobname) {
            $logfile =~ s/\$SLURM_ARRAY_TASK_ID/$job_id/g;
          }
          my $last_line = `tail -n 1 $logfile`;
          if ($last_line =~ m/status 0$/ && (-M $logfile) < 0) {
            # if the last line of $logfile ended with "status 0" and
            # $logfile is newer than this program [(-M $logfile) gives the
            # time elapsed between file modification and the start of this
            # program], then we assume the program really finished OK,
            # and maybe something is up with the file system.
            print STDERR "**$0: syncfile $f was not created but job seems\n" .
              "**to have finished OK.  Probably your file-system has problems.\n" .
              "**This is just a warning.\n";
            last;
          } else {
            chop $last_line;
            print STDERR "$0: Error: Job $sge_job_id seems to no longer exists:\n" .
              "'squeue -j $sge_job_id' returned error code $squeue_status and said:\n" .
              "  $squeue_output\n" .
              "Syncfile $f does not exist, meaning that the job did not finish.\n" .
              "Log is in $logfile. Last line '$last_line' does not end in 'status 0'.\n" .
              "Possible reasons:\n" .
              "  a) Exceeded time limit? -> Use more jobs!\n" .
              "  b) Shutdown/Frozen machine? -> Run again! squeue:\n";
            system("squeue -j $sge_job_id");
            exit(1);
          }
        } elsif ($ret != 0) {
          print STDERR "$0: Warning: squeue command returned status $ret (squeue -j $sge_job_id,$!)\n";
        }
      }
    }
  }
  my $all_syncfiles = join(" ", @syncfiles);
  system("rm $all_syncfiles 2>/dev/null");
}

# OK, at this point we are synced; we know the job is done.
# But we don't know about its exit status.  We'll look at $logfile for this.
# First work out an array @logfiles of file-locations we need to
# read (just one, unless it's an array job).
my @logfiles = ();
if (!defined $jobname) { # not an array job.
  push @logfiles, $logfile;
} else {
  for (my $jobid = $jobstart; $jobid <= $jobend; $jobid++) {
    my $l = $logfile;
    $l =~ s/\$SLURM_ARRAY_TASK_ID/$jobid/g;
    push @logfiles, $l;
  }
}

my $num_failed = 0;
my $status = 1;
foreach my $l (@logfiles) {
  my @wait_times = (0.1, 0.2, 0.2, 0.3, 0.5, 0.5, 1.0, 2.0, 5.0, 5.0, 5.0, 10.0, 25.0);
  for (my $iter = 0; $iter <= @wait_times; $iter++) {
    my $line = `tail -10 $l 2>/dev/null`; # Note: although this line should be the last
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
    if (defined $jobname) { $logfile =~ s/\$SLURM_TASK_ARRAY_ID/$jobstart/g; }
    print STDERR "$0: job failed with status $status, log is in $logfile\n";
    if ($logfile =~ m/JOB/) {
      print STDERR "$0: probably you forgot to put JOB=1:\$nj in your script.\n";
    }
  } else {
    if (defined $jobname) { $logfile =~ s/\$SLURM_ARRAY_TASK_ID/*/g; }
    my $numjobs = 1 + $jobend - $jobstart;
    print STDERR "$0: $num_failed / $numjobs failed, log is in $logfile\n";
  }
  exit(1);
}
