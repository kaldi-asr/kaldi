# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

# At BUT use:
export train_cmd="queue.pl -q all.q@@stable -l ram_free=1G,mem_free=1G"
export decode_cmd="queue.pl -q all.q@@stable -l ram_free=2G,mem_free=2G"
export highmem_cmd="queue.pl -q all.q@@stable -l ram_free=2G,mem_free=2G"

# On Eddie use:
# export train_cmd="queue.pl -P inf_hcrc_cstr_general"
# export decode_cmd="queue.pl -P inf_hcrc_cstr_general"
# export highmem_cmd="queue.pl -P inf_hcrc_cstr_general -pe memory-2G 2"

# To run locally, use:
# export train_cmd=run.pl
# export decode_cmd=run.pl
# export highmem_cmd=run.pl
