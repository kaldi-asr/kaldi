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
#export cuda_cmd=run.pl


#b) BUT cluster options
export train_cmd="queue.pl -q all.q@blade[01][0126789][123456789] -l ram_free=2500M,mem_free=2500M,matylda5=0.5"
export decode_cmd="queue.pl -q all.q@blade[01][0126789][123456789] -l ram_free=3000M,mem_free=3000M,matylda5=0.1"
export mkgraph_cmd="queue.pl -q all.q@blade[01][0126789][123456789] -l ram_free=4G,mem_free=4G,matylda5=3"
export cuda_cmd="queue.pl -q long.q@pcspeech-gpu,long.q@dellgpu*,long.q@pco203-0[0124] -l gpu=1" 

#c) run locally...
#export train_cmd=run.pl
#export decode_cmd=run.pl
#export cuda_cmd=run.pl
#export mkgraph_cmd=run.pl
