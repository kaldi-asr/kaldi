# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

# Run locally:
#export train_cmd=run.pl
#export decode_cmd=run.pl
#export cuda_cmd=run.pl

# JHU cluster:
export train_cmd="queue.pl"
export decode_cmd="queue.pl --mem 4G"
# the use of cuda_cmd is deprecated, used only in 'nnet1',
export cuda_cmd="queue.pl --gpu 1"

host=$(hostname -f)
if [ ${host#*.} == "fit.vutbr.cz" ]; then
  queue_conf=$HOME/queue_conf/default.conf # see example /homes/kazi/iveselyk/queue_conf/default.conf,
  export train_cmd="queue.pl --config $queue_conf --mem 2G --matylda 0.2"
  export decode_cmd="queue.pl --config $queue_conf --mem 3G --matylda 0.1"
  export cuda_cmd="queue.pl --config $queue_conf --gpu 1 --mem 10G --tmp 40G"
elif [ ${host#*.} == "cm.cluster" ]; then
  # MARCC bluecrab cluster:
  export train_cmd="slurm.pl --time 4:00:00 "
  export decode_cmd="slurm.pl --mem 4G --time 4:00:00 "
  export cuda_cmd="slurm.pl --gpu 1"
fi
