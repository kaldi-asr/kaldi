# THIS IS A COPY FROM RAYMOND: /share/spandh.ami1/lid/eval/lre15/kaldi/scripts.swb1_r2/cmd.sh

# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

## SUBMIT
# export train_cmd="queue.pl -j y -l qp=NORMAL -l memq=4000 -l h_vmem=4G -P MINI"
export train_cmd="queue.pl -j y -l qp=NORMAL -l memq=4000 -l h_vmem=4G -l hostname='node1|node20|node21|node22|node23|node24|node11|node25|node26' -P MINI "		# Because of GLIBC problem during upgrading
export train_online2_cmd="queue.pl -j y -l qp=NORMAL -l hostname='node1|node20|node21|node23|node24|node25|node26' -l memq=5000 -l h_vmem=5G -P MINI" 
export train_online2_multi_cmd="queue.pl -j y -q multi2 -l qp=MULTI -l memq=5000 -l h_vmem=5G -P MINI" 
# export train_online2_splice_cmd="queue.pl -j y -l qp=GPU -l memq=5000 -l h_vmem=5G -P MINI" 
export train_online2_splice_cmd="queue.pl -j y -l qp=GPU -l memq=5000  -P MINI"
export train_dnn_cmd="queue.pl -j y -q multi2 -l qp=MULTI -l memq=2000 -l h_vmem=2G -P MINI" 
# export decode_cmd="queue.pl -j y -l qp=MULTI -l memq=3000 -l h_vmem=3G -P MINI"
export decode_cmd="queue.pl -j y -l qp=MULTI -l memq=3000 -l h_vmem=3G -l hostname='node1|node20|node21|node22|node23|node24|node25|node26' -P MINI"		# Because of GLIBC problem during upgrading
# export decode_large_cmd="queue.pl -j y -l qp=MULTI -l memq=5000 -l h_vmem=5G -P MINI"
export decode_large_cmd="queue.pl -j y -l qp=MULTI -l memq=5000 -l h_vmem=5G -l hostname='node1|node20|node21|node22|node23|node24|node25|node26' -P MINI"	# Because of GLIBC problem during upgrading


# use 8G*(3+1)threads=32G
# export score_cmd="queue.pl -j y -l qp=NORMAL -l memq=2000 -l h_vmem=2G -P MINI"
export score_cmd="queue.pl -j y -l qp=NORMAL -l memq=2000 -l h_vmem=2G -l hostname='node1|node20|node21|node22|node23|node24|node11|node25|node26' -P MINI"		# Because of GLIBC problem during upgrading
# export decode_dnn_cmd="queue.pl -j y -q multi3 -q multi1 -q multi2 -q multi4 -l qp=MULTI -l memq=15000 -l h_vmem=15G -P MINI" 
# export decode_dnn_cmd="queue.pl -j y -q multi3 -q multi1 -q multi2 -q multi4 -l qp=MULTI -l memq=4000 -l h_vmem=4G -P MINI" 
# export decode_dnn_cmd="queue.pl -j y -l memq=4000 -l qp=GPU -l h_vmem=4G -P MINI"		# Yulan Liu, 7 Mar 2016
export decode_dnn_cmd="queue.pl -j y -l memq=10000 -l qp=GPU -P MINI"               # Yulan Liu, 31 Oct 2016
# export decode_dnn_cmd="queue.pl -j y -l qp=NORMAL -l hostname='node20.minigrid.dcs.shef.ac.uk|node21.minigrid.dcs.shef.ac.uk' -l memq=30000 -l h_vmem=30G -P MINI" 
# export score_dnn_cmd="queue.pl -j y -l qp=NORMAL -l memq=2000 -l h_vmem=2G -P MINI"
export score_dnn_cmd="queue.pl -j y -l qp=NORMAL -l memq=2000 -l h_vmem=2G -l hostname='node1|node20|node21|node22|node23|node24|node11|node25|node26' -P MINI"	# Because of GLIBC problem during upgrading
# export align_dnn_cmd="queue.pl -j y -l qp=NORMAL -l hostname='node20|node21|node23|node24' -l memq=2000 -l h_vmem=2G -P MINI"
export align_dnn_cmd="queue.pl -j y -q multi1 -q multi2 -q multi4 -l qp=MULTI -l memq=2000 -l h_vmem=2G -P MINI"
export denlat_dnn_cmd="queue.pl -j y -q multi1 -q multi2 -q multi4 -l qp=MULTI -l memq=2000 -l h_vmem=2G -P MINI"
export mkgraph_cmd="queue.pl -j y -l qp=NORMAL -l memq=50000 -l h_vmem=50G -P MINI"
export big_memory_cmd="queue.pl -j y -l qp=NORMAL -l memq=8000 -l h_vmem=8G -P MINI"
# export cuda_cmd="queue.pl -j y -l hostname='node20|node21|node22|node24|node25|node26' -l qp=GPU -l memq=75000 -l h_vmem=75G -P MINI"
export cuda_cmd="queue.pl -j y -l hostname='node20|node21|node22|node24|node25|node26' -l qp=GPU -l memq=75000 -P MINI"
# cuda_cmd tried 4G, 10G, 30G
# export cuda_large_cmd="queue.pl -j y -l hostname='node20|node21' -l qp=GPU -l memq=150000 -l h_vmem=150G -P MINI"
# export cuda_large_cmd="queue.pl -j y -l hostname='node20|node21' -l qp=GPU -l memq=50000 -P MINI"
# export cuda_large_cmd="queue.pl -j y -l hostname='node20|node21|node22|node23|node24' -l qp=GPU -l memq=20000 -P MINI"		# This was from Raymond
# cuda_cmd tried 4G, 10G, 30G, (was 75G, 150G)
# export cuda_large_cmd="queue.pl -j y -l hostname='node20|node21|node22|node24|node25|node26' -l qp=GPU -l memq=100000 -l h_vmem=100G -P MINI"			# This is to tune to SDM and MDM MPE DNN training
export cuda_large_cmd="queue.pl -j y -l hostname='node20|node21|node22|node24|node25|node26' -l qp=GPU -l memq=100000 -P MINI"                   # This is to tune to SDM and MDM MPE DNN training



# LOCAL
# train_cmd="queue.pl -l arch=*64"
# decode_cmd="queue.pl -l arch=*64"

# cuda_cmd is used for nnet1 scripts e.g. local/run_dnn.sh, but
# in the nnet2 scripts e.g. local/run_nnet2.sh, this is not
# used and we append options to train_cmd.
# cuda_cmd="queue.pl -l arch=*64 -l gpu=1"

#train_cmd="run.pl"
# with run.pl we do training locally.  Note: for jobs on smallish subsets,
# it's way faster to run on a single machine with a handful of CPUs, as
# you avoid the latency of starting GridEngine jobs.


# Added by Yulan on 2 Mar 2016
export LC_ALL=C
export highmem_cmd="queue.pl -j y -l qp=NORMAL -l memq=20000 -l h_vmem=20G -l hostname='node1|node20|node21|node22|node23|node24|node11' -P MINI"		# This command is required by latest Kaldi script, mkgraph.sh

