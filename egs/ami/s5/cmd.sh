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

export train_cmd="queue.pl --mem 1G"
export decode_cmd="queue.pl --mem 2G"
# the use of cuda_cmd is deprecated but it is sometimes still used in nnet1
# scripts.
export cuda_cmd="queue.pl --gpu 1 --mem 20G"

# the rest of this file is present for historical reasons.
# In general it's best to rely on conf/queue.conf for cluster-specific
# configuration.

# On Eddie use:
#export train_cmd="queue.pl -P inf_hcrc_cstr_nst -l h_rt=08:00:00"
#export decode_cmd="queue.pl -P inf_hcrc_cstr_nst  -l h_rt=05:00:00 -pe memory-2G 4"
#export highmem_cmd="queue.pl -P inf_hcrc_cstr_nst -l h_rt=05:00:00 -pe memory-2G 4"
#export scoring_cmd="queue.pl -P inf_hcrc_cstr_nst  -l h_rt=00:20:00"

#<<<<<<< HEAD
# JSALT2015 workshop, cluster AWS-EC2, (setup from Vijay)
export train_cmd="queue.pl -l arch=*64* --mem 1G"
export decode_cmd="queue.pl -l arch=*64* --mem 2G"
export highmem_cmd="queue.pl -l arch=*64* --mem 4G"
export scoring_cmd="queue.pl -l arch=*64*"
export cuda_cmd="queue.pl --gpu 1 -l mem_free=20G,ram_free=20G"
export cuda_mem_cmd="queue.pl --gpu 1 -l mem_free=42G,ram_free=42G"
export cntk_decode_cmd="queue.pl -l arch=*64* --mem 1G -pe smp 2"

# To run locally, use:
#export train_cmd=run.pl
#export decode_cmd=run.pl
#export highmem_cmd=run.pl
#export cuda_cmd=run.pl

#=======
#>>>>>>> 6c7c0170812a1f7dfb5c09c078787e79ee72333a
#if [ "$(hostname -d)" == "fit.vutbr.cz" ]; then
#  # BUT cluster:
#  queue="all.q@@blade,all.q@@speech"
#  gpu_queue="long.q@@gpu"
#  storage="matylda5"
#  export train_cmd="queue.pl -q $queue -l ram_free=1.5G,mem_free=1.5G,${storage}=1"
#  export decode_cmd="queue.pl -q $queue -l ram_free=2.5G,mem_free=2.5G,${storage}=0.5"
#  export cuda_cmd="queue.pl -q $gpu_queue -l gpu=1"
#fi

