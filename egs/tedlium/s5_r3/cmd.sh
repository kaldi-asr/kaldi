# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

# Run locally:
#export train_cmd=run.pl
#export decode_cmd=run.pl

# JHU cluster (or most clusters using GridEngine, with a suitable
# conf/queue.conf).
export train_cmd="queue.pl"
export decode_cmd="queue.pl --mem 4G"
