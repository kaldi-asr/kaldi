# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

#a) JHU cluster options
#export train_cmd="queue.pl -l arch=*64"
#export decode_cmd="queue.pl -l arch=*64,mem_free=2G,ram_free=2G"
#export mkgraph_cmd="queue.pl -l arch=*64,ram_free=4G,mem_free=4G"
#export cuda_cmd=run.pl


if [ "$(hostname -d)" == "clsp.jhu.edu" ]; then
  export train_cmd="queue.pl -l arch=*64*"
  export decode_cmd="queue.pl -l arch=*64* --mem 3G"
  export cuda_cmd="queue.pl -l gpu=1"
elif [ "$(hostname -d)" == "fit.vutbr.cz" ]; then
  #b) BUT cluster options
  queue="all.q@@blade,all.q@@speech"
  gpu_queue="long.q@@gpu"
  storage="matylda5"
  export train_cmd="queue.pl -q $queue -l ram_free=1.5G,mem_free=1.5G,${storage}=0.5"
  export decode_cmd="queue.pl -q $queue -l ram_free=2.5G,mem_free=2.5G,${storage}=0.1"
  export cuda_cmd="queue.pl -q $gpu_queue -l gpu=1" 
else
  echo "$0: you need to define options for your cluster."
  exit 1;
fi

#c) run locally...
#export train_cmd=run.pl
#export decode_cmd=run.pl
#export cuda_cmd=run.pl
#export mkgraph_cmd=run.pl
