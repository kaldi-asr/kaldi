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
export train_cmd="queue.pl -l arch=*64*"
export decode_cmd="queue.pl -l arch=*64* -l ram_free=4G,mem_free=4G"
export cuda_cmd="queue.pl -l arch=*64*,gpu=1 -q g.q"

# BUT cluster:
host=$(hostname)
if [ ${host#*.} == "fit.vutbr.cz" ]; then
  queue="all.q@@blade,all.q@@speech"
  gpu_queue="long.q@supergpu*,long.q@dellgpu*,long.q@pcspeech-gpu,long.q@pcgpu*"
  storage="matylda5"
  export train_cmd="queue.pl -q $queue -l ram_free=1500M,mem_free=1500M,${storage}=1"
  export decode_cmd="queue.pl -q $queue -l ram_free=2500M,mem_free=2500M,${storage}=0.5"
  export cuda_cmd="queue.pl -q $gpu_queue -l gpu=1" 
fi
