#!/bin/bash

# you can change these commands to just run.pl to make them run
# locally, but in that case you should change the num-jobs to
# the #cpus on your machine or fewer.
decode_cmd="queue.pl -q all.q@@blade -l ram_free=1200M,mem_free=1200M"
train_cmd="queue.pl -q all.q@@blade -l ram_free=700M,mem_free=700M"
cuda_cmd="queue.pl -q long.q@@pco203 -l gpu=1"
mkgraph_cmd="queue.pl -q all.q@@servers -l ram_free=4G,mem_free=4G"

# put the scripts to path
source path.sh



######################################################
###    HERE START THE DEEP NETWORK EXPERIMENTS     ###
######################################################

# A.
# we will use si84 data, and 2500 leaves

# pre-train the MLPs
numleaves=2500
dir=exp/tri2a-${numleaves}_deep_nnet_pretrain/
ali=exp/tri2a-${numleaves}_ali
$cuda_cmd $dir/_pretrain_nnet.log \
  steps/pretrain_nnet_dev_alter_rbm_xent.sh --lrate 0.002 data/train_si84 data/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1;

# finetune the MLPs
hidL=$(seq -w 1 10)
for hid in ${hidL[@]}; do
( #do this in subshell
  dir=exp/tri2a-${numleaves}_deep_nnet_pretrain_finetune_hid$hid/
  ali=exp/tri2a-${numleaves}_ali
  $cuda_cmd $dir/_finetune_nnet.log \
    steps/train_nnet_dev_MLPINIT.sh --lrate 0.001 \
    --mlp-init exp/tri2a-${numleaves}_deep_nnet_pretrain/nnet/hid${hid}b_nnet.xent \
    data/train_si84 data/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || { touch $dir/.error; exit 1; }
  #decode
  $decode_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr exp/tri2a-${numleaves} $dir/graph_tgpr || { touch $dir/.error; exit 1; }
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 || { touch $dir/.error; exit 1; }
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 || { touch $dir/.error; exit 1; }
)&
done

wait
if [ $(ls  exp/tri2a-${numleaves}_deep_nnet_finetune*/.error | wc -l) -gt 0 ]; then
  echo "Error occured in the Deep Network finetuning"
fi 



# B.
# we will use si284 data, and 2500 leaves

# pre-train the MLPs
numleaves=2500
dir=exp/tri2a-${numleaves}_deep_nnet_pretrain_si284/
ali=exp/tri2a-${numleaves}_ali
$cuda_cmd $dir/_pretrain_nnet.log \
  steps/pretrain_nnet_dev_alter_rbm_xent.sh --lrate 0.002 data/train_si284 data/test_dev93 data/lang ${ali}_si284 ${ali}_dev93 $dir || exit 1;

# finetune the MLPs
hidL=$(seq -w 1 10)
for hid in ${hidL[@]}; do
( #do this in subshell
  dir=exp/tri2a-${numleaves}_deep_nnet_pretrain_si284_finetune_hid$hid/
  ali=exp/tri2a-${numleaves}_ali
  $cuda_cmd $dir/_finetune_nnet.log \
    steps/train_nnet_dev_MLPINIT.sh --lrate 0.001 \
    --mlp-init exp/tri2a-${numleaves}_deep_nnet_pretrain/nnet/hid${hid}b_nnet.xent \
    data/train_si284 data/test_dev93 data/lang ${ali}_si284 ${ali}_dev93 $dir || { touch $dir/.error; exit 1; }
  #decode
  $decode_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr exp/tri2a-${numleaves} $dir/graph_tgpr || { touch $dir/.error; exit 1; }
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_dev93 $dir/decode_tgpr_dev93 || { touch $dir/.error; exit 1; }
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data/test_eval92 $dir/decode_tgpr_eval92 || { touch $dir/.error; exit 1; }
)&
done

wait
if [ $(ls  exp/tri2a-${numleaves}_deep_nnet_finetune*/.error | wc -l) -gt 0 ]; then
  echo "Error occured in the Deep Network finetuning"
fi 


