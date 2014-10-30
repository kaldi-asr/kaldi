# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

#train_cmd='queue.pl -q all.q@a03.clsp.jhu.edu,all.q@a06.clsp.jhu.edu,all.q@a05.clsp.jhu.edu,all.q@v01.clsp.jhu.edu,all.q@a10.clsp.jhu.edu,all.q@a04.clsp.jhu.edu,all.q@a13.clsp.jhu.edu,all.q@a11.clsp.jhu.edu -l arch=*64'
#decode_cmd='queue.pl -q all.q@a03.clsp.jhu.edu,all.q@a06.clsp.jhu.edu,all.q@a05.clsp.jhu.edu,all.q@v01.clsp.jhu.edu,all.q@a10.clsp.jhu.edu,all.q@a04.clsp.jhu.edu,all.q@a13.clsp.jhu.edu,all.q@a11.clsp.jhu.edu -l arch=*64'
train_cmd="queue.pl -l arch=*64"
decode_cmd="queue.pl -l arch=*64"
#train_cmd="run.pl"
# Do training locally.  Note: for jobs on smallish subsets,
# it's way faster to run on a single machine with a handful of CPUs, as
# you avoid the latency of starting GridEngine jobs.



