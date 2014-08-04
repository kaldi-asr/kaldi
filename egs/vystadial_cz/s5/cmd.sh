# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

#export train_cmd="queue.pl -q all.q@a*.clsp.jhu.edu"
#export decode_cmd="queue.pl -q all.q@a*.clsp.jhu.edu"
# export train_cmd="queue.pl -l mf=5g"
# export decode_cmd="queue.pl -l mf=5g"
export train_cmd="queue.pl -l arch=*64*"
export decode_cmd="queue.pl -l arch=*64*"

# The number of parallel jobs to be started for some parts of the recipe
# Make sure you have enough resources(CPUs and RAM) to accomodate this number of jobs
njobs=20

# If you have no GridEngine you can do:
#export train_cmd=run.pl
#export decode_cmd=run.pl
#njobs=2
