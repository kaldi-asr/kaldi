# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

train_cmd="queue.pl -l arch=*64"
decode_cmd="queue.pl -l arch=*64"

# cuda_cmd is used for nnet1 scripts e.g. local/run_dnn.sh, but
# in the nnet2 scripts e.g. local/run_nnet2.sh, this is not
# used and we append options to train_cmd.
cuda_cmd="queue.pl -l arch=*64 -l gpu=1"

#train_cmd="run.pl"
# with run.pl we do training locally.  Note: for jobs on smallish subsets,
# it's way faster to run on a single machine with a handful of CPUs, as
# you avoid the latency of starting GridEngine jobs.


# BUT cluster:
host=$(hostname -f)
if [ ${host#*.} == "fit.vutbr.cz" ]; then
  queue="all.q@@blade,all.q@@speech"
  gpu_queue="long.q@supergpu*,long.q@dellgpu*,long.q@pcspeech-gpu,long.q@pcgpu*"
  storage="matylda5"
  export train_cmd="queue.pl -q $queue -l ram_free=1500M,mem_free=1500M,${storage}=1"
  export decode_cmd="queue.pl -q $queue -l ram_free=2500M,mem_free=2500M,${storage}=0.5"
  export cuda_cmd="queue.pl -q $gpu_queue -l gpu=1" 
fi
