# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

#a) JHU cluster options
export train_cmd="queue.pl -l arch=*64"
export decode_cmd="queue.pl -l arch=*64 --mem 2G"
export mkgraph_cmd="queue.pl -l arch=*64 --mem 4G"
export big_memory_cmd="queue.pl -l arch=*64 --mem 8G"
export cuda_cmd="queue.pl -l gpu=1"

#b) run it locally...
#export train_cmd=run.pl
#export decode_cmd=run.pl
#export cuda_cmd=run.pl
#export mkgraph_cmd=run.pl

#c) BUT cluster:
if [ "$(hostname -d)" == "fit.vutbr.cz" ]; then
  queue="all.q@@blade,all.q@@speech"
  gpu_queue="long.q@@gpu"
  storage="matylda5"
  export train_cmd="queue.pl -q $queue -l ram_free=1.5G,mem_free=1.5G,${storage}=1"
  export decode_cmd="queue.pl -q $queue -l ram_free=2.5G,mem_free=2.5G,${storage}=0.5"
  export cuda_cmd="queue.pl -q $gpu_queue -l gpu=1"
fi
