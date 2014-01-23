# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

decode_cmd="queue.pl -q all.q@blade* -l mem_free=2G,ram_free=2G"
train_cmd="queue.pl -q all.q@blade* -l mem_free=2G,ram_free=2G"
cuda_cmd="queue.pl -q long.q@dellgpu*,long.q@pcspeech-gpu -l gpu=1"

#train_cmd="queue.pl -l arch=*64"
#decode_cmd="queue.pl -l arch=*64"
#train_cmd="run.pl"
# Do training locally.  Note: for jobs on smallish subsets,
# it's way faster to run on a single machine with a handful of CPUs, as
# you avoid the latency of starting GridEngine jobs.



