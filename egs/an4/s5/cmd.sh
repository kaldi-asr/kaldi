# you can change cmd.sh depending on what type of queue you are using.
# If you have no queueing system and want to run on a local machine, you
# can change all instances 'queue.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  queue.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different queues are configured differently, with different
# queue names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/queue.conf to match your queue's configuration.  Search for
# conf/queue.conf in http://kaldi-asr.org/doc/queue.html for more information,
# or search for the string 'default_config' in utils/queue.pl or utils/slurm.pl.

export train_cmd=queue.pl
export decode_cmd=queue.pl
export mkgraph_cmd=queue.pl
export cuda_cmd="queue.pl --gpu 1"


# The rest of this file is here for historical reasons.  For cluster-specific
# configuration it's generally better to use conf/queue.conf, see
# http://kaldi-asr.org/doc/queue.html.

# BUT cluster:
if [ "$(hostname -d)" == "fit.vutbr.cz" ]; then
  queue="all.q@@blade,all.q@@speech"
  gpu_queue="long.q@@gpu"
  storage="matylda5"
  export train_cmd="queue.pl -q $queue -l ram_free=1500M,mem_free=1500M,${storage}=1"
  export decode_cmd="queue.pl -q $queue -l ram_free=2500M,mem_free=2500M,${storage}=0.5"
  export cuda_cmd="queue.pl -q $gpu_queue -l gpu=1"
fi
