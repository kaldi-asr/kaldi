# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

export train_cmd="slurm.pl --mem 6G --config conf/slurm.conf"
export decode_cmd="slurm.pl  --config conf/slurm.conf"
export cuda_cmd="slurm.pl gpu --mem 6G --gpu 2 --config conf/slurm.conf"
