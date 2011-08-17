#!/bin/bash

if [ $# != 3 ]; then
  echo $0 SGE_OPTIONS COMMAND FULL_PATH_WORKDIR; exit 1;
fi

SGE_OPTIONS=$1
COMMAND=$2
WORKDIR=$3

if [ ! -e $WORKDIR ]; then
  mkdir -p $WORKDIR
fi

{ echo "#!/bin/bash"
  echo "cd ${PWD}"
  echo hostname
  echo $COMMAND
  echo echo SGE_END
  echo echo
  echo sleep 10
} > $WORKDIR/sge_task.sh

#run in sge or locally
( echo "SGE_LOG" > $WORKDIR/sge_task.out; tail --pid=$$ -s 1 -F $WORKDIR/sge_task.out) &
qsub $SGE_OPTIONS -o $WORKDIR/sge_task.out -e $WORKDIR/sge_task.out -sync yes $WORKDIR/sge_task.sh || $COMMAND


