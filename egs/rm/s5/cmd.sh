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



