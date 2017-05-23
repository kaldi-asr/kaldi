# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

# Run locally:
#export train_cmd=run.pl
#export decode_cmd=run.pl

# JHU cluster (or most clusters using GridEngine, with a suitable
# conf/queue.conf).
export train_cmd="queue.pl"
export decode_cmd="queue.pl --mem 4G"

host=$(hostname -f)
if [ ${host#*.} == "fit.vutbr.cz" ]; then
  # BUT cluster:
  queue="all.q@@blade,all.q@@speech"
  gpu_queue="long.q@@gpu"
  storage="matylda5"
  export train_cmd="queue.pl -q $queue -l ram_free=1500M,mem_free=1500M,${storage}=1"
  export decode_cmd="queue.pl -q $queue -l ram_free=2500M,mem_free=2500M,${storage}=0.5"
elif [ ${host#*.} == "cm.cluster" ]; then
  # MARCC bluecrab cluster:
  export train_cmd="slurm.pl --time 4:00:00 "
  export decode_cmd="slurm.pl --mem 4G --time 4:00:00 "
fi
