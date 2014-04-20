# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

#a) JHU cluster options
#export train_cmd="queue.pl -l arch=*64"
#export decode_cmd="queue.pl -l arch=*64,mem_free=2G,ram_free=2G"
#export mkgraph_cmd="queue.pl -l arch=*64,ram_free=4G,mem_free=4G"
#export big_memory_cmd="queue.pl -l arch=*64,ram_free=8G,mem_free=8G"
#export cuda_cmd="queue.pl -l gpu=1"



#b) BUT cluster options
#export train_cmd="queue.pl -q all.q@@blade -l ram_free=1200M,mem_free=1200M"
#export decode_cmd="queue.pl -q all.q@@blade -l ram_free=1700M,mem_free=1700M"
#export decodebig_cmd="queue.pl -q all.q@@blade -l ram_free=4G,mem_free=4G"

#export cuda_cmd="queue.pl -q long.q@@pco203 -l gpu=1"
#export cuda_cmd="queue.pl -q long.q@pcspeech-gpu"
#export mkgraph_cmd="queue.pl -q all.q@@servers -l ram_free=4G,mem_free=4G"

#c) run it locally...
export train_cmd=run.pl
export decode_cmd=run.pl
export cuda_cmd=run.pl
export mkgraph_cmd=run.pl
