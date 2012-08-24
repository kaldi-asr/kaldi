#!/bin/bash

# you can change these commands to just run.pl to make them run
# locally, but in that case you should change the num-jobs to
# the #cpus on your machine or fewer.

#1)BUT
decode_cmd="queue.pl -q all.q@@blade -l ram_free=1200M,mem_free=1200M"
train_cmd="queue.pl -q all.q@@blade -l ram_free=700M,mem_free=700M"
cuda_cmd="queue.pl -q long.q@@pco203 -l gpu=1"
#cuda_cmd="queue.pl -q long.q@pcspeech-gpu"
mkgraph_cmd="queue.pl -q all.q@@servers -l ram_free=4G,mem_free=4G"

#2)JHU
#export train_cmd="queue.pl -q all.q@a*.clsp.jhu.edu -S /bin/bash"
#export decode_cmd="queue.pl -q all.q@a*.clsp.jhu.edu -S /bin/bash"

#3)LOCAL
#export train_cmd=run.pl
#export decode_cmd=run.pl
#export cuda_cmd=run.pl
#export mkgraph_cmd=run.pl



# put the scripts to path
source path.sh



######################################################
###    HERE START THE DEEP NETWORK EXPERIMENTS     ###
######################################################

# A.
# we will use si84 data, and 2500 leaves

# pre-train the MLPs
dir=exp/tri2a_deep_nnet_pretrain/
ali=exp/tri2a_ali
$cuda_cmd $dir/_pretrain_nnet.log \
  steps/pretrain_nnet_alter_rbm_xent.sh --lrate 0.002 --nn-depth 10 --nn-dimhid 1024 \
  data-fbank/train_si84 data-fbank/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1;

# finetune the MLPs
pretrain=$dir
for hid in $(seq -f '%02g' 1 10); do
( #do this in subshell
  dir=exp/tri2a_deep_nnet_pretrain_finetune_hid$hid/
  ali=exp/tri2a_ali
  $cuda_cmd $dir/_finetune_nnet.log \
    steps/train_nnet.sh --lrate 0.001 \
    --mlp-init $pretrain/nnet/hid${hid}b_nnet.xent \
    data-fbank/train_si84 data-fbank/test_dev93 data/lang ${ali}_si84 ${ali}_dev93 $dir || exit 1;
  #decode
  $mkgraph_cmd $dir/_mkgraph.log scripts/mkgraph.sh data/lang_test_tgpr $dir $dir/graph_tgpr || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data-fbank/test_dev93 $dir/decode_tgpr_dev93 || exit 1;
  scripts/decode.sh --cmd "$decode_cmd" --opts "--acoustic-scale 0.12" steps/decode_nnet.sh $dir/graph_tgpr data-fbank/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
)&
done
wait



