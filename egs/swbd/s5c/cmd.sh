# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

# Default opts,
export train_cmd="queue.pl -l arch=*64*"
export decode_cmd="queue.pl -l arch=*64* --mem 4G"
export cuda_cmd=run.pl # Run on local machine,
export mkgraph_cmd="queue.pl -l arch=*64* --mem 4G"

# BUT options,
if [ "$(hostname -d)" == "fit.vutbr.cz" ]; then
  # BUT cluster:
  queue="all.q@@blade,all.q@@speech"
  gpu_queue="long.q@supergpu*,long.q@dellgpu*,long.q@pcspeech-gpu,long.q@pcgpu*"
  storage="matylda5"
  export train_cmd="queue.pl -q $queue -l ram_free=1.5G,mem_free=1.5G,${storage}=0.25"
  export decode_cmd="queue.pl -q $queue -l ram_free=2.5G,mem_free=2.5G,${storage}=0.1"
  export cuda_cmd="queue.pl -q $gpu_queue -l gpu=1"
fi 

