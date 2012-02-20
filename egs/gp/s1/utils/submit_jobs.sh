#!/bin/bash -u

# Copyright 2012  Arnab Ghoshal

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

set -o errexit

function error_exit () {
  echo -e "$@" >&2; exit 1;
}

function readposint () {
  local retval=`expr "X$1" : '[^=]*=\(.*\)'`;
  retval=${retval#0*}  # Strip any leading 0's
  [[ "$retval" =~ ^[1-9][0-9]*$ ]] \
    || error_exit "Argument \"$retval\" not a positive integer."
  echo $retval
}

PROG=`basename $0`;
usage="Usage: $PROG [options] --log=logfile command\n
Runs the supplied command and redirect the stdout & stderr to logfile.\n
With the --qcmd option, the command is submitted to a grid engine.\n
Any 'TASK_ID' in logfile or command is replaced with job number or \$SGE_TASK_ID (for SGE).\n\n
Required arguments:\n
  --log=FILE\tOutput of command redirected to this file.\n\n
Options:\n
  --njobs=INT\tNumber of jobs to run (default=1). Assumes split data exists.\n
  --qcmd=STRING\tCommand for submitting a job to a grid engine (e.g. qsub) including switches.\n
";

if [ $# -lt 2 ]; then
  error_exit $usage;
fi

NJOBS=1     # Default number of jobs
QCMD=""     # No grid usage by default
while [ $# -gt 1 ]; do
  case "${1# *}" in  # ${1# *} strips any leading spaces from the arguments
  --help) echo -e $usage; exit 0 ;;
  --qcmd=*)
  QCMD=`expr "X$1" : '[^=]*=\(.*\)'`; shift ;;
  --njobs=*)
  NJOBS=`readposint $1`; shift ;;
  --log=*)
  LOGF=`expr "X$1" : '[^=]*=\(.*\)'`; shift ;;
  -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
  '')  shift ;;  # Handle any empty arguments
  *)   break ;;  # interpreted as the command to execute
  esac
done

logfile_base=`basename $LOGF .log`
logfile_dir=`dirname $LOGF`
mkdir -p $logfile_dir;

# Now, parse the command to execute
exec_cmd="";
while [ $# -gt 0 ]; do
  case "$1" in
  *\"*) exec_cmd=$exec_cmd"'''$1''' "; shift ;;
  *\ *) exec_cmd=$exec_cmd"\"$1\" "; shift ;;
     *) exec_cmd=$exec_cmd"$1 "; shift ;;
  esac
done

function run_locally {
  rm -f $logfile_dir/.error;
  for n in `seq 1 $NJOBS`; do
    local this_logfile=${logfile_base//TASK_ID/$n}
    this_logfile=$logfile_dir"/"$this_logfile".log"
    local this_command=${exec_cmd//TASK_ID/$n}
    ( echo -e "# Command:\n# $this_command";
      echo "# Running on: "`hostname`;
      echo "# Started at: "`date`;
      eval $this_command || touch $logfile_dir/.error
      echo "# Finished at: "`date` ) >> $this_logfile 2>&1 &
  done
  wait;
  [ -f $logfile_dir/.error ] && { rm -f $logfile_dir/.error; \
      error_exit "One (or more) locally run jobs failed."; }
  exit 0;
}

function run_on_grid {
  local this_logfile=${logfile_base//TASK_ID/\$SGE_TASK_ID}
  this_logfile=$logfile_dir"/"$this_logfile".log"
  # If log files are in a separate 'log' directory, create the job submission
  # scripts one level up.
  local qdir=${logfile_dir/%log/q}
  mkdir -p $qdir
  local qlog=$qdir/queue.log
  local this_command=${exec_cmd//TASK_ID/\$SGE_TASK_ID}
  local run_this=$qdir"/"${logfile_base//TASK_ID/}".sh"
  run_this=${run_this//../.}
  printf "#!/bin/bash\n#\$ -S /bin/bash\n#\$ -V -cwd -j y\n" > $run_this
  { printf "set -e\n";
    printf "{ cd %s\n  . path.sh\n  echo Running on: \`hostname\`\n" "$PWD";
    printf "  echo Started at: \`date\`\n  $this_command\n  ret=\$\?\n";
    printf "  echo Finished at: \`date\`\n} >& %s\nexit \$ret\n" "$this_logfile"
    printf "# Submitted with:\n"
    printf "# $QCMD -sync y -o $qlog -t 1-$NJOBS $run_this >> $qlog 2>&1\n"
  } >> $run_this
  $QCMD -sync y -o $qlog -t 1-${NJOBS} $run_this >> $qlog 2>&1
  exit $?
}

if [ -z "$QCMD" ]; then
  run_locally;
else
  run_on_grid;
fi

