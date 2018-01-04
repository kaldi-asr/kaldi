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
export tfrnnlm_cmd="queue.pl -l hostname=b*"  # this is specific to the CLSP grid
# we limit to certain machines because TF binaries compiled from C++ take advantage
# of machine architectures for optimization

# this is the cmd we use for, specifically on the CLSP grid,
# training RNNLM with tensorflow. I have gpu here but it actually depends on the versions
# installed - in tools/extras/install_tensorflow_py.sh we could install either
# the GPU or CPU version; if the CPU version is installed, we should remove the
# gpu 1 option here so as to not waste GPUs
# as for the CUDA_VISIBLE_DEVICES variable this is because TensorFlow would
# automatically utilize all resources, and in this case multiple GPUs if it is
# present on the machine. This option is only necessary if you're using GPUs and
# want to limit the job only on the GPU that you reserve

# the use of cuda_cmd is deprecated, used only in 'nnet1',
export cuda_cmd="queue.pl --gpu 1 --mem 20G"

if [[ "$(hostname -f)" == "*.fit.vutbr.cz" ]]; then
  queue_conf=$HOME/queue_conf/default.conf # see example /homes/kazi/iveselyk/queue_conf/default.conf,
  export train_cmd="queue.pl --config $queue_conf --mem 2G --matylda 0.2"
  export decode_cmd="queue.pl --config $queue_conf --mem 3G --matylda 0.1"
  export cuda_cmd="queue.pl --config $queue_conf --gpu 1 --mem 10G --tmp 40G"
fi

# On Eddie use:
#export train_cmd="queue.pl -P inf_hcrc_cstr_nst -l h_rt=08:00:00"
#export decode_cmd="queue.pl -P inf_hcrc_cstr_nst  -l h_rt=05:00:00 -pe memory-2G 4"
#export highmem_cmd="queue.pl -P inf_hcrc_cstr_nst -l h_rt=05:00:00 -pe memory-2G 4"
#export scoring_cmd="queue.pl -P inf_hcrc_cstr_nst  -l h_rt=00:20:00"

