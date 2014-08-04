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

# BUT cluster:
export train_cmd="queue.pl -q all.q@blade[01][0126789][123456789] -l ram_free=2500M,mem_free=2500M,matylda5=0.5"
export decode_cmd="queue.pl -q all.q@blade[01][0126789][123456789] -l ram_free=3000M,mem_free=3000M,matylda5=0.1"
export cuda_cmd="queue.pl -q long.q@pcspeech-gpu -l gpu=1" 

