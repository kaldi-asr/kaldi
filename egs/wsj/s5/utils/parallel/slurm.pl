#!/usr/bin/env perl

# SPDX-License-Identifier: Apache-2.0
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).
#           2014  Vimal Manohar (Johns Hopkins University)
#           2015  Johns Hopkins University (Yenda Trmal <jtrmal@gmail.com>>)
#           2020  Kirill 'kkm' Katsnelson <kkm@pobox.com>

use strict;
use warnings;

use File::Basename;
use IPC::Open2;

# This is a cleaned-up version of old slurm.pl. The conifguration file format
# and the default configration values are shared by the two scripts; refer to
# the latter for the description. NOTE: this includes the default partition
# names 'shared' (for non-GPU jobs) and 'gpu' (for CUDA jobs), so you'll likely
# need the config anyway.
#
# Notable differences:
# * All command-line switches with a single '-' are reported as errors (-sync,
#   -V, ...). These were carried over from SGE/PBS qsub, and have no equivalent
#   or, worse, an entirely different semantics (e.g., '-V') in Slurm's squeue.
#   No Kaldi recipes rely on switch passing to squeue as of the moment of this
#   rewrite. Consequently, all transformations of generic Kaldi switches to
#   Slurm switches are performed solely through the config mappings.
# * This version relies on Slurm machinery to wait for and get the completion
#   status of the job (i.e., all jobs are launched synchronously).
#
# Shortcomings (carried over from queue.pl and slurm.pl):
# * the --gpu switch mapped through the config usually specifies a queue name
#   (a partition in Slurm parlance). This makes it impossible to combine a
#   switch selecting a partition with the --gpu to override the parititon.
#   Similarly, all swithces in the 'command' declaration are impossible to
#   override. If this proves necessary, the switch-mapping will have to be
#   reworked (e.g., Kaldi switch maps to an array of switches, and of the
#   repeating ones, the textually last matching one in the config wins). Not
#   doing that at the moment.

# When 1, prints sbatch commands. When 2, also prints batch file contents.
# Note that the debug level 2 is very verbose.
my $debug = 0;

sub usage() {
  print STDERR << "EOF";
Usage: $0 [options] [JOB=1[:n]] log-file command-line [arguments...]
e.g.:  $0 foo.log echo baz
  (which will echo \"baz\", with stdout and stderr directed to foo.log)
or: $0 foo.log echo bar \| sed s/bar/baz/
  (which is an example of using a pipe; you can provide other escaped bash
  constructs)
or: $0 JOB=1:10 foo.JOB.log echo JOB
  (which illustrates the mechanism to submit parallel jobs; note, you can use
  another string other than JOB)
Options:
  --config <config-file> (default: use internal defaults, see source)
  --mem <mem-requirement> (e.g. 2G, 50M, 2048K, or 2000000; default: no limit)
  --num-threads <num-threads> (default: 1)
  --max-jobs-run <num-jobs> (default: no limit)
  --gpu <0|1> (default: 0)
EOF
  exit 2;
}

sub fatal   { print STDERR "$0: fatal: ",   @_, "\n"; exit 1; }
sub warning { print STDERR "$0: warning: ", @_, "\n"; }
sub info    { print STDERR "$0: info: ", @_, "\n" if $debug; }
sub debug   { print STDERR "$0: DEBUG: ",   @_, "\n" if $debug > 1; }

#------------------------------------------------------------------------------#
# Parse command line.
#---------------------

my %cli_options;

# Parsed job array spec; all are 'undef's if not seen.
my $jobvar;    # 'JOB' of 'JOB=2:42'
my $jobstart;  # '2'   of 'JOB=2:42'
my $jobend;    # '42'  of 'JOB=2:42'

my $original_command_line=join(" ", @ARGV);  # Printed if --debug 2, later.

PARSE_ARGS:
while (@ARGV >= 2) {
  my $arg = $ARGV[0];

  # Switches.
  if ($arg =~ m/^-/) {
    my $switch = shift @ARGV;
    fatal "invalid switch '$arg'; single-dash switches are unsupported!"
        unless $arg =~ m/^--/;

    # '--' ends this script switch parse.
    last PARSE_ARGS if $arg eq '--';

    # Convert CLI option to variable name by removing '--' from the switch and
    # replacing any '-' with a '_' ('--max-jobs-run' => 'max_jobs_run').
    $switch =~ s/^--//;
    $switch =~ s/-/_/g;

    my $swarg = shift @ARGV;
    $cli_options{$switch} = $swarg;

    $debug = $swarg if ($switch eq 'debug');

    next PARSE_ARGS;
  }

  # Job spec, JOB=1:42 or JOB=1
  if ($arg =~ m/^([\pL_][\pL\d_]*) = (\d+)(?: :(\d+) )?$/ax) {
    fatal "only one job spec is allowed: '$arg'" if $jobvar;
    $jobvar = $1;
    $jobstart = $2;
    $jobend = $3 // $2;  # Treat 'X=42' as if it were 'X=42:42'.
    fatal "invalid job range '$arg': start > end" if $jobstart > $jobend;
    fatal "invalid job range '$arg': must be positive" if $jobstart == 0;
    shift;
    next PARSE_ARGS;
  }

  $arg =~ m/.=.*:/ and
      warning "suspicious argument '$arg', malformed job spec?";

  last PARSE_ARGS;
}

my $logfile = shift @ARGV;
if ($jobvar && $jobstart < $jobend && $logfile !~ m/$jobvar/) {
  fatal "you are trying to run a parallel job but ".
      "you are putting the output into just one log file '$logfile'";
}

debug $original_command_line;

# Not enough arguments.
usage() if @ARGV < 2;

#------------------------------------------------------------------------------#
# Parse configration.
#---------------------

#TODO(kkm): slurm.pl is used currently in 3 recopes: formosa, tedlium r2, mgb5.
# Possibly we do not need the defaults at all? However, '--no-kill' and
# '--hint=compute_bound' should always be added to the 'sbatch' command if this
# defult config is eliminated.
my $default_config_spec = q {
# Default configuration. You must specify a 'command', and at the least rules
# for the 'gpu', 'mem' and 'num_threads' options, as Kaldi scripts sometimes add
# them. The rest is for your use, e. g. to tune the commands in recipe's cmd.sh.
command sbatch --no-kill --hint=compute_bound --export=PATH #--ntasks-per-node=1
option time=* --time=$0
option mem=* --mem-per-cpu=$0
option mem=0          # Do not add any options for '--mem 0'.
default num_threads=1
option num_threads=* --cpus-per-task=$0
default gpu=0
option gpu=0 #-p shared
option gpu=* -p xa-gpu --gres=cuda:$0
# note: the --max-jobs-run option is supported as a special case
# by slurm.pl and you don't need to handle it in the config file.
};

# Here the configuration options specified by the user on the command line
# (e.g. --mem 2G) are converted to Slurm sbatch options system as defined in
# the config file. (e.g. if the config file has the line
#
#    "option mem=* mem_per_cpu=$0,num_threads=2"
#
# and the user has specified '--mem 2G' on the command line, the options passed
# to sbatch would be "--mem-per-cpu=2G --num-threads=2". The config commands
# in the last part of the string thus are native sbatch switches, reformatted,
# and $0 is substituted for the config-provided value for the '*'.

my %config_rules = ();
my $sbatch_cmd;
my $config_file;

if (exists $cli_options{"config"}) {
  # Open specified config file.
  $config_file = $cli_options{"config"};
  open (CONFIG, "<", $config_file) or
      fatal "cannot open configuration file $config_file: $!";
} else {
  # Open internal config string as file.
  $config_file = "(default config)";
  open (CONFIG, "<", \$default_config_spec) or die "internal error: $!";
}
delete $cli_options{"config"};  # Done with the config switch.

PARSE_CONFIG:
while(<CONFIG>) {
  chomp;
  my $line = s/\s*#.*//gr or next; # Skip over empty and comment-only lines.
  my ($keyword, $pattern, $arg) = split(' ', $line, 3);

  $keyword eq 'command' and do {
    $sbatch_cmd = join(' ', $pattern, $arg // ());
    next
  };
  goto CONFIG_ERROR unless $pattern ne '';

  my ($option, $value) = split('=', $pattern, 2);
  goto CONFIG_ERROR unless $option ne '' && $value ne '';
  $keyword eq 'option' and do {
    if ($value eq '*') {
      $value = undef;
      $arg =~ m/\$0/ or warning "$config_file:$.: the line '$_' does not ".
          'contain the substitution variable $0';
    }
    $config_rules{($option,$value // ())} = $arg // ''; next
  };

  $keyword eq 'default' and do {
    # Add option unless already provided by the user.
    $cli_options{$option} //= $value; next
  };

CONFIG_ERROR:
  fatal "invalid line $config_file:$.: '$_'"
};
close(CONFIG);

$sbatch_cmd or
    fatal "'$config_file' does not contain the directive \"command\"";

#------------------------------------------------------------------------------#
# Evaluate Slurm batch options.
#---------------------

# Slurm uses special syntax for maximum simultaneous jobs in array, not
# handled by the rules.
my $max_jobs_run;

for my $option (keys %cli_options) {
  my $value = $cli_options{$option};

  if ($option eq "max_jobs_run") {
    if ($jobvar) {
      $max_jobs_run = $value;
    } else {
      warning "option '--max-jobs-run' ignored since this is not an array task";
    }
  } elsif (exists $config_rules{($option,$value)}) {
    # Preset option-value pair, e. g., 'gpu=0'
    $sbatch_cmd .= " " . $config_rules{($option,$value)};
  } elsif (exists $config_rules{($option)}) {
    # Option with a substitution token '$0', e. g., 'gpu=*'
    $sbatch_cmd .= " " . $config_rules{($option)} =~ s/\$0/$value/gr;
  } else {
    fatal "no rule in '$config_file' matches option/value '$option=$value'";
  }
}

#------------------------------------------------------------------------------#
# Work out the command; quote escaping is done here.
#---------------------
# Note: the rules for escaping stuff are worked out pretty arbitrarily, based on
# what we want it to do. Some things that we pass as arguments to $0, such as
# "|", we want to be interpreted by bash, so we don't escape them. Other things,
# such as archive specifiers like 'ark:gunzip -c foo.gz|', we want to be passed,
# in quotes, to the Kaldi program. Our heuristic is that stuff with spaces in
# should be quoted. This doesn't always work.

# TODO(kkm): I do not like this. Can we just quote all single quotes in the
#            string (sub each ' with '"'"'), and always single-quote arguments,
#            so we are safe against shell expansions entirely? Or is there code
#            that relies on this?

my $cmd = "";
for (@ARGV) {
  if (/^\S+$/) { $cmd .= $_ . " "; } # If string contains no spaces, take as-is.
  elsif (/\"/) { $cmd .= "'$_' "; } # else if has dbl-quotes, use single
  else { $cmd .= "\"$_\" "; }  # else use double.
}
$cmd =~ s/$jobvar/\$\{SLURM_ARRAY_TASK_ID\}/g if $jobvar;

# Create log directory.
my $logdir = dirname($logfile);
system("mkdir -p $logdir") == 0 or exit 1;  # message is printed by mkdir.

#------------------------------------------------------------------------------#
# Compose sbatch command and our script
#---------------------

$sbatch_cmd .= " --parsable --wait";

# Work out job name from log file basename by cutting everything after the last
# dot. If nothing left, well, use the default ('sbatch'). This must not happen.
my $jobname = basename($logfile) =~ s/\.[^.]*$//r;
# remove '.JOB' from jobname, if array job.
$jobname =~ s/\.$jobvar//g if $jobvar;
$sbatch_cmd .= " --job-name=$jobname" if $jobname;

# Add array spec if array job.
if ($jobvar) {
  $sbatch_cmd .= " --array=${jobstart}-${jobend}";
  $max_jobs_run and $max_jobs_run > 0 and $sbatch_cmd .= "%${max_jobs_run}";
}

# Overwrite log file.
$logfile =~ s/$jobvar/%a/g if $jobvar; # The '%a' token is for array index.
$sbatch_cmd .= " --output=$logfile --open-mode=truncate";

# num_threads is for the "Accounting" line only. Also note we drop oom_adj and
# renice to normal priority. For some reason, slurmstepd runs the script at its
# own nice valie (-10), while we want to be killed by the OOM killer before it
# kills slurmstepd, which takes the node longer to recover.
my $num_threads = $cli_options{'num_threads'} // 1;
my $batch_script = join('',
'#!/bin/bash', q{
echo "# Running on $(hostname)"
echo "# Started at $(date "${TIME_STYLE:---}")"
printenv | grep ^SLURM | sort | while read; do echo '#' "$REPLY"; done},
# This is a way of echoing the command into a comment in the log file,
# without having to escape things like "|" and quote characters.
    qq{
cat <<EOF_xeC9eg
# $cmd
EOF_xeC9eg}, q{
if [[ ${CUDA_VISIBLE_DEVICES-} = NoDevFiles ]]; then
  echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting.
  unset CUDA_VISIBLE_DEVICES
fi
time1=$(date +%s)
set -x
renice 0 $$ >/dev/null
echo 5 >/proc/self/oom_adj}, qq{
$cmd}, q{
ret=$?
set +x
sync
time2=$(date +%s)
echo "# Accounting: begin_time=$time1"
echo "# Accounting: end_time=$time2"
echo "# Accounting: time=$((time2-time1)) }, qq{threads=$num_threads"}, q{
echo "# Finished at $(date "${TIME_STYLE:---}") with status $ret"
exit $ret
# submitted with:}, qq{
## $sbatch_cmd
} );

#TODO(kkm): Save script to file for diagnostic maybe? Slurm holds it for 30
# minutes on job failure, enough for diafnostics.
debug "Submitting batch script:\n------\n${batch_script}------";

#------------------------------------------------------------------------------#
# Submit job using sbatch
#----------------------
#TODO(kkm): Handle signals and cancel jobs? I do often break scripts...

# open2 prints the command and dies on failure, no need for the 'or die ...'.
my ($rh_sbatch, $wh_sbatch);
my $pid_sbatch = open2($rh_sbatch, $wh_sbatch, $sbatch_cmd);

# Ignore a possible failure here, we'll know of it anyway.
print $wh_sbatch $batch_script;
close $wh_sbatch;

# Receive job ID, either '1234' or 'cluster:1234'. If not returned, assume that
# the submission process failed, otherwise status would reflect a job failure.
my $slurm_jobid = <$rh_sbatch>;
if (defined $slurm_jobid) {
  chomp($slurm_jobid);
  info "Submitted job $slurm_jobid with: $sbatch_cmd";
}

# This blocks until the job completes, because '--wait'.
waitpid($pid_sbatch, 0);
my $rc = $?;
$rc >>= 8 if $rc > 255;

# sbatch should have explained what happened by this point.
defined($slurm_jobid) or
    fatal "sbatch command failed with exit code $rc: $sbatch_cmd";

# TODO(kkm): If controller cannot accept a job because of a temporary failure,
# sbatch waits for a hard-coded loop of MAX_RETRIES, defined to 15, incrementing
# sleep by 1 every time, 105s total.  We can loop here retrying the submission
# for quite longer. Never happens in the cloud, but physical fail-over may take
# a longer time.

exit 0 if $rc == 0;

$logfile =~ s/%a/*/g;
#TODO(kkm): Invoke 'scontrol show -o job=$slurm_jobid' to count failed jobs?
# This make sense only for arrays. I did see failures of individual jobs.
#
fatal "Job $slurm_jobid failed with status $rc. Check $logfile";

#------------------------------------------------------------------------------#
