# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

# On Eddie use:
#export train_cmd="queue.pl -P inf_hcrc_cstr_nst -l h_rt=08:00:00"
#export decode_cmd="queue.pl -P inf_hcrc_cstr_nst  -l h_rt=05:00:00 -pe memory-2G 4"
#export highmem_cmd="queue.pl -P inf_hcrc_cstr_nst -l h_rt=05:00:00 -pe memory-2G 4"
#export scoring_cmd="queue.pl -P inf_hcrc_cstr_nst  -l h_rt=00:20:00"

# To run locally, use:
export train_cmd=run.pl
export decode_cmd=run.pl
export highmem_cmd=run.pl
export cuda_cmd=run.pl

host=$(hostname -f)
if [ ${host#*.} == "fit.vutbr.cz" ]; then
  # BUT cluster:
  queue="all.q@@blade,all.q@@speech"
  gpu_queue="long.q@supergpu*,long.q@dellgpu*,long.q@pcspeech-gpu,long.q@pcgpu*"
  storage="matylda5"
  export train_cmd="queue.pl -q $queue -l ram_free=1500M,mem_free=1500M,${storage}=1"
  export decode_cmd="queue.pl -q $queue -l ram_free=2500M,mem_free=2500M,${storage}=0.5"
  export cuda_cmd="queue.pl -q $gpu_queue -l gpu=1"
fi 

